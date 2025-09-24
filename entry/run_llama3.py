import sys
import argparse
sys.path.append("../")
sys.path.append("../utils")
sys.path.append('../pybind/build')

from utils.prof_marker import prof_marker
from utils.frontend import requestManager
from utils.util_functions import prepare_weight
from transformers import AutoTokenizer
from utils.input_test import prefill_context
import torch


# from models.llama3_KVCacheTorch import Pipeline
from models.llama3_FlashinferKVCache import Pipeline

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-l", "--load_hf_weight", action="store_true", help="Load weights from huggingface")

args = arg_parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# request_queue = []
# request_manager = requestManager(args.trace_path, "meta-llama/Meta-Llama-3-8B-Instruct")
# request_manager.read_request()
# request_manager.release_request()
# # print(request_manager.available_request_queue)

# new_input_ids = []
# for req in request_manager.available_request_queue:
#     print("req.idx: ", req.req_idx)
#     print("req.prompt: ", req.prompt)
#     print("req.output_len: ", req.output_len)
#     new_input_ids.append((req.req_idx, req.prompt))
# print("new_input_ids: ", new_input_ids)



# weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
weight_map_wzr = "/root/nanoflow/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
# weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
# weight_map_amd_kan = "/work1/kasikci/kanzhu/models/llama3-8b"
# weight_map_yi = "/app/llama3-8b"
if args.load_hf_weight:
    pipeline_weight_list = [
        (i, f"cuda:{i}", Pipeline()) for i in range(1)
    ]
    prepare_weight(pipeline_weight_list, weight_map_wzr)

pipeline = Pipeline()
pipeline.init(weight_map_wzr, cached=True)

# torch.cuda.empty_cache()
# device = torch.cuda.current_device()
# reserved_memory = torch.cuda.memory_reserved(device)
# print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")
# pipeline.config()
def test_performance():
    seq_len = 512
    global_batch_size = 256
    decode_batch_size = 64
    prefill_batch_size = global_batch_size - decode_batch_size

    prefill_context_ids = tokenizer.encode(prefill_context) # which length is 1912.

    prefill_input_ids = prefill_context_ids[:seq_len]
    output_strings = {}
    # initialize the reqs for first 384 requests
    decode_inputs = []
    for i in range(decode_batch_size):
        input = [(i, prefill_input_ids.copy())]
        pipeline.update(input)
        output_strings[i] = prefill_input_ids.copy()
        new_tokens = pipeline.run()
        for _, new_token in new_tokens:
            output_strings[i].extend(new_token)
        decode_inputs.extend(new_tokens)
        print("new_tokens: ", new_tokens)

    # prepare for the testing configuration
    output_strings[decode_batch_size] = prefill_context_ids[:prefill_batch_size].copy()
    decode_inputs.extend([(decode_batch_size, prefill_context_ids[:prefill_batch_size].copy())])
    pipeline.update(decode_inputs, decode_batch_size, profile_result_path="../auto_search/8B_search_result_large_btz.json", use_cuda_graph=True, use_nano_split=True)
    # pipeline.update(decode_inputs, decode_batch_size)

    for i in range(decode_batch_size, decode_batch_size + 20):
        print("Cycle: ", i - decode_batch_size)
        next_prefill_idx = i + 1
        new_tokens = pipeline.run()
        with prof_marker(f"after_execute_step_4"):
            for req_idx, new_token in new_tokens:
                output_strings[req_idx].extend(new_token)
        # print("new_tokens: ", new_tokens)
        with prof_marker(f"after_execute_step_5"):
            new_tokens = new_tokens[:-1]
            decode_batchsize = len(new_tokens)
            assert decode_batchsize == decode_batch_size
        with prof_marker(f"after_execute_step_6"):
            output_strings[next_prefill_idx] = prefill_context_ids[:prefill_batch_size].copy()
        with prof_marker(f"after_execute_step_7"):
            new_tokens.extend([(next_prefill_idx, prefill_context_ids[:prefill_batch_size].copy())])
        with prof_marker(f"after_execute_step_8"):
            pipeline.update(new_tokens, decode_batchsize, profile_result_path="../auto_search/8B_search_result_large_btz.json", use_cuda_graph=True, use_nano_split=True)
            # pipeline.update(new_tokens, decode_batchsize)

    output_text = tokenizer.batch_decode(list(output_strings.values())[:1], skip_special_tokens=True)
    print(output_text)

def test_correctness(use_kv_cache=True):
    # input_strings = ["Hi, who are you?"]
    # input_strings = ["Hi, who are you?", "What's the weather today?"]
    input_string = "Hi, who are you?"
    # input_strings = [ "The university of washington is located in" for _ in range(16)]
    input_ids = tokenizer.encode(input_string)
    # print(input_ids)
    special_inputs_0 = [(0, input_ids.copy()), (1, input_ids.copy())]
    special_inputs_1 = [(2, input_ids.copy()), (3, input_ids.copy())]
    output_strings = {}
    for idx, tensor in special_inputs_0:
        output_strings[idx] = tensor

    for idx, tensor in special_inputs_1:
        output_strings[idx] = tensor

    pipeline.update(special_inputs_0)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    decode_batchsize = len(new_tokens)
    assert decode_batchsize == 2
    
    # print("new_tokens: ", new_tokens)
    if use_kv_cache:
        new_tokens.extend(special_inputs_1)
        pipeline.update(new_tokens, decode_batchsize)
    else:
        new_tokens = [(0, output_strings[0]), (1, output_strings[1])]
        new_tokens.extend(special_inputs_1)
        pipeline.update(new_tokens, 0)

    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    decode_batchsize = len(new_tokens)
    assert decode_batchsize == 4
    # print("new_tokens: ", new_tokens)

    if use_kv_cache:
        pipeline.update(new_tokens, decode_batchsize)
    else:
        new_tokens = [(i, output_strings[i]) for i in range(4)]
        pipeline.update(new_tokens, 0)

    for i in range(20):
        print("Cycle: ", i)
        new_tokens = pipeline.run()
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        decode_batchsize = len(new_tokens)
        assert decode_batchsize == 4
        # print("new_tokens: ", new_tokens)
        if use_kv_cache:
            pipeline.update(new_tokens, decode_batchsize)
        else:
            new_tokens = [(i, output_strings[i]) for i in range(4)]
            pipeline.update(new_tokens, 0)

    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)

def test_one_cycle():
    input_string = "Hi, who are you?"
    # input_strings = [ "The university of washington is located in" for _ in range(16)]
    input_ids = tokenizer.encode(input_string)
    special_inputs_0 = [(0, input_ids.copy()), (1, input_ids.copy())]
    output_strings = {}
    for idx, tensor in special_inputs_0:
        output_strings[idx] = tensor

    pipeline.update(special_inputs_0)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    
    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)

def profile_one_cycle():
    prefill_context_ids = tokenizer.encode(prefill_context) # which length is 1912.
    
    # 添加调试信息
    print("初始化 profile data...")
    pipeline.init_profile_data()
    
    # 检查 SM 配置
    print(f"Total SM count: {pipeline.total_sm}")
    print(f"SM counts list: {pipeline.sm_counts}")
    
    '''# 检查 gen_embedding 的实现映射
    if hasattr(pipeline, 'gen_embedding'):
        print(f"GenEmbedding impl_map keys: {list(pipeline.gen_embedding.impl_map.keys()) if hasattr(pipeline.gen_embedding, 'impl_map') else 'No impl_map'}")
    else:
        print("No gen_embedding found in pipeline")'''
    
    stream_names = [ f"TEST_{i}" for i in range(len(pipeline.sm_counts)) ] + ["TEST_TOTAL"]
    print("\nAvailable streams and their SM counts:")
    for stream_name in stream_names:
        if stream_name in pipeline.profile_streams:
            stream, sm_count = pipeline.profile_streams[stream_name]
            print(f"Stream {stream_name}: SM count = {sm_count}")
        else:
            print(f"Warning: Stream {stream_name} not found in profile_streams")
    
    for stream_name in stream_names:
        print(f"\nProcessing Stream: {stream_name}")
        if stream_name not in pipeline.profile_streams:
            print(f"Skipping stream {stream_name} as it's not configured")
            continue
        pipeline.reset()

        # test for prefill
        # total_batch_sizes = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
        # total_batch_sizes = [128, 256, 384, 512, 640]   #小批量测试        ---512报错out of memory
        total_batch_sizes = [128, 256, 384, 512, 640]   #小批量测试        ---512报错out of memory
        # total_batch_sizes = [1024]
        for idx, total_batch_size in enumerate(total_batch_sizes):
            input = [(idx, prefill_context_ids[:total_batch_size].copy())]
            
            # 打印输入信息
            print(f"\nPreparing to process batch:")
            print(f"- Input length: {len(input)}")
            print(f"- Input[0] shape: {len(input[0][1])}")
            print(f"- Total batch size: {total_batch_size}")
            
            try:
                # 确保输入数据在正确的设备上
                if isinstance(input[0][1], torch.Tensor):
                    print(f"- Input device: {input[0][1].device}")
                else:
                    print("- Input is not a tensor")
                
                pipeline.update(input, is_profile=True, stream_name=stream_name)
                pipeline.profile_run()
            except Exception as e:
                print(f"\nError in batch processing:")
                print(f"- Error type: {type(e).__name__}")
                print(f"- Error message: {str(e)}")
                raise

        # test for decode
        # total_batch_sizes = [128, 256, 384, 512, 640]
        # total_batch_sizes = [128, 256, 384]  #小批量测试
        total_batch_sizes = [128, 256]  #小批量测试
        # total_batch_sizes = [384]
        # prepare the decode inputs for a special input_length
        # input_length = 1024
        # input_length = 512  #测试用较小的长度，搭配384的batch_size out of memory了
        input_length = 256  #测试用更小的长度
        output_length = 0
        prefill_input_ids = prefill_context_ids[:input_length]

        for total_batch_size in total_batch_sizes:
            decode_inputs = []
            pipeline.reset()
            for i in range(total_batch_size):
                input = [(i, prefill_input_ids.copy())]
                pipeline.update(input, is_profile=True)
                new_tokens = pipeline.run()
                decode_inputs.extend(new_tokens)
                print("new_tokens: ", new_tokens)
                print("total_batch_size: ", total_batch_size)

            # decode profiling from input_length to input_length + output_length
            for i in range(output_length + 1):
                print("Cycle: ", i)
                pipeline.update(decode_inputs, total_batch_size, is_profile=True, stream_name=stream_name)
                if i % 128 == 0:
                    pipeline.profile_run()
                    
    print("All profiling data has been collected.")

# test_correctness()
# test_correctness(use_kv_cache=False)
# test_performance()
# test_one_cycle()
profile_one_cycle()