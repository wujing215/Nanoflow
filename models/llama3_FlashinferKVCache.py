import copy
import json
from typing import Any, Optional
import torch

from utils.green_ctx import split_device_green_ctx_by_sm_count
from operations.operation_base import NanoOpInfo, Operations, Operation_Layer
from operations.activation.silu import Activation
from operations.embedding.embedding import GenEmbedding
from operations.globalOp.globalOp import GlobalInput, GlobalOutput
from operations.gemm.gemm_N_parallel import GEMM_N_Parallel
from operations.norm.rmsnorm import LayerNorm
from operations.sampling.max_sampling import Sampling
from operations.rope.rope_flashinfer import RopeAppendFlashinfer
from operations.attention.llamaAttention_flashinfer import DecAttnFlashinfer, PFAttnFlashinfer
from operations.virtualOp.virtual_ops import Copy, Redist
from kvcache.kv import KVCacheNone, DistKVPool, BatchedDistKVCache
from core.weightManager import WeightManager
from core.bufferAllocate import BufferAllocator
from core.executor import Executor
from core.nanobatchSplit import split_nanobatch
from core.categoryType import CategoryType
from utils.prof_marker import prof_marker

class Pipeline():
    def __init__(self):
        # Set parameters as instance variables.
        self.pipeline_name = "Llama3-8B"
        self.num_kv_heads = 8
        self.num_qo_heads = 32
        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads
        self.head_dim = 128
        self.vocab_size = 128256
        self.hidden_dim = 4096
        self.intermediate_dim = 14 * 1024
        self.global_batch_size: Optional[int] = None
        self.decode_batch_size: Optional[int] = None
        self.num_layers = 32
        self.layer_list = [i for i in range(self.num_layers)]
        self.page_size = 16
        self.device = "cuda:0"
        self.profile_dir = f"../profile_data/{self.pipeline_name}"
        self.profile_result: dict[str, Any] | None = None
        self.categories = [CategoryType.COMP, CategoryType.MEM]

        self.buffer_fixed: bool = False
        self.is_auto_search_enabled: bool = False
        self.is_cuda_graph_enabled: bool = False
        self.plan_cuda_graph: bool = False

    def set_device(self, rank, device):
        pass

    def init(self, weight_path, cached=False):
        self.init_streams()
        self.init_external_data()
        self.init_operations()
        self.init_category()
        self.init_dependency()
        self.init_set_shape()
        self.init_set_weight(weight_path, cached)

    def init_streams(self):
        self.main_stream = torch.cuda.Stream()
        # 获取设备实际的 SM 数量
        max_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.total_sm = min(max_sms, 132)  # 使用实际 SM 数量和预期值中的较小值
        print(f"Using total SM count: {self.total_sm} (device has {max_sms} SMs)")
        
        # 生成不超过实际 SM 数量的 sm_counts 列表
        self.sm_counts = [
            i for i in range(8, min(128, max_sms), 8) # Assuming SM counts are in increments of 8
        ]
        # [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
        num_sm_counts = len(self.sm_counts)
        self.streams_test = {
            "GEMM_Test": (torch.cuda.Stream(), self.total_sm),
        }

        self.streams: dict[CategoryType, dict[int, tuple[torch._C.Stream, int]]] = {}
        for category in self.categories:##
            self.streams[category] = {}

            '''for i in range((num_sm_counts + 1) // 2):
                sm_count_1 = self.sm_counts[i]
                sm_count_2 = self.sm_counts[num_sm_counts - 1 - i]
                print(f"Creating green context streams for SM counts: {sm_count_1}, {sm_count_2}")

                (stream_1, stream_2, _), _ = split_device_green_ctx_by_sm_count(
                    torch.device(self.device),
                    [sm_count_1, sm_count_2]
                )
                self.streams[category][sm_count_1] = (stream_1, sm_count_1)
                self.streams[category][sm_count_2] = (stream_2, sm_count_2)
            self.streams[category][self.total_sm] = (torch.cuda.Stream(), self.total_sm)'''

            print("!!! WARNING: Bypassing green_ctx creation, creating standard CUDA streams for all SM counts. !!!")

            # Iterate through each required SM count and create a standard stream
            for sm_count in self.sm_counts:
                stream = torch.cuda.Stream()
                self.streams[category][sm_count] = (stream, sm_count)

            # The original last line is still correct and needed
            self.streams[category][self.total_sm] = (torch.cuda.Stream(), self.total_sm)

        # Create green context streams for testing
        self.profile_streams: dict[str, tuple[torch._C.Stream, int]] = {}
        '''for i in range((num_sm_counts + 1) // 2):
            sm_count_1 = self.sm_counts[i]
            sm_count_2 = self.sm_counts[num_sm_counts - 1 - i]
            print(f"Creating green context streams for SM counts: {sm_count_1}, {sm_count_2}")

            (stream_1, stream_2, _), _ = split_device_green_ctx_by_sm_count(
                torch.device(self.device),
                [sm_count_1, sm_count_2]
            )
            self.profile_streams[f"TEST_{i}"] = (stream_1, sm_count_1)
            self.profile_streams[f"TEST_{num_sm_counts - 1 - i}"] = (stream_2, sm_count_2)
        self.profile_streams[f"TEST_TOTAL"] = (torch.cuda.Stream(), self.total_sm)'''

        print("!!! 警告：正在绕过 green_ctx 创建，为性能分析创建标准的 CUDA 流。!!!")

        for i in range((num_sm_counts + 1) // 2):
            sm_count_1 = self.sm_counts[i]
            sm_count_2 = self.sm_counts[num_sm_counts - 1 - i]
            
            # --- 这是关键的改动 ---
            # 我们不再调用有问题的函数，而是创建两个标准的流。
            stream_1 = torch.cuda.Stream()
            stream_2 = torch.cuda.Stream()
            # ----------------------------

            # 其余用于存储流的原始逻辑保持不变。
            self.profile_streams[f"TEST_{i}"] = (stream_1, sm_count_1)
            self.profile_streams[f"TEST_{num_sm_counts - 1 - i}"] = (stream_2, sm_count_2)

        # 原始代码的最后一行也是正确的，应当保留。
        self.profile_streams[f"TEST_TOTAL"] = (torch.cuda.Stream(), self.total_sm)


    def init_external_data(self):
        # self.kv_pool = DistKVPool(self.num_layers, self.num_kv_heads, self.head_dim, 2048, self.page_size, 1, self.device)
        # self.kv_pool = DistKVPool(self.num_layers, self.num_kv_heads, self.head_dim, 2048* 26, self.page_size, 1, self.device)
        self.kv_pool = DistKVPool(self.num_layers, self.num_kv_heads, self.head_dim, 10240, self.page_size, 1, self.device)
        self.kv_cache = BatchedDistKVCache(self.kv_pool)
    
    def reset(self):
        # reset kv cache
        self.kv_cache.reset()

        # reset batch size and decode batch size
        self.global_batch_size = None
        self.decode_batch_size = None

    def init_operations(self):
        self.global_input    = GlobalInput("GlobalInput", self.device).first_only()
        self.global_input_layers = self.global_input.expand_layer(self.layer_list)

        self.gen_embedding  = GenEmbedding("GenEmbedding", self.device).setWeightName("model.embed_tokens.weight").first_only()
        self.gen_embedding_layers = self.gen_embedding.expand_layer(self.layer_list)

        self.layerNormAttn   = LayerNorm("LayerNormAttn", self.device).setWeightName("model.layers.{layer}.input_layernorm.weight")
        self.layerNormAttn_layers = self.layerNormAttn.expand_layer(self.layer_list)

        self.kqv             = GEMM_N_Parallel("KQV", self.device).setWeightName([
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            "model.layers.{layer}.self_attn.q_proj.weight"
        ])
        self.kqv_layers = self.kqv.expand_layer(self.layer_list)

        self.ropeAppend      = RopeAppendFlashinfer("RopeAppend", self.device)
        self.ropeAppend.externals["KVCache"] = self.kv_cache
        self.ropeAppend_layers = self.ropeAppend.expand_layer(self.layer_list)


        self.decAttn         = DecAttnFlashinfer("DecAttn", self.device)
        self.decAttn.externals["KVCache"] = self.kv_cache
        self.decAttn_layers = self.decAttn.expand_layer(self.layer_list)

        self.pfAttn          = PFAttnFlashinfer("PFAttn", self.device)
        self.pfAttn.externals["KVCache"] = self.kv_cache
        self.pfAttn_layers = self.pfAttn.expand_layer(self.layer_list)

        self.o               = GEMM_N_Parallel("O", self.device, bias=True).setWeightName("model.layers.{layer}.self_attn.o_proj.weight")
        self.o_layers = self.o.expand_layer(self.layer_list)

        self.layerNormFFN    = LayerNorm("LayerNormFFN", self.device).setWeightName("model.layers.{layer}.post_attention_layernorm.weight")
        self.layerNormFFN_layers = self.layerNormFFN.expand_layer(self.layer_list)

        self.ug              = GEMM_N_Parallel("UG", self.device).setWeightName([
            "model.layers.{layer}.mlp.up_proj.weight",
            "model.layers.{layer}.mlp.gate_proj.weight"
        ])
        self.ug_layers = self.ug.expand_layer(self.layer_list)

        self.activation      = Activation("Activation", self.device)
        self.activation_layers = self.activation.expand_layer(self.layer_list)

        self.d               = GEMM_N_Parallel("D", self.device, bias=True).setWeightName("model.layers.{layer}.mlp.down_proj.weight")
        self.d_layers = self.d.expand_layer(self.layer_list)


        self.getLogits       = GEMM_N_Parallel("GetLogits", self.device).setWeightName("lm_head.weight").last_only()
        self.getLogits_layers = self.getLogits.expand_layer(self.layer_list)


        self.modelLayerNorm  = LayerNorm("ModelLayerNorm", self.device).setWeightName("model.norm.weight").last_only()
        self.modelLayerNorm_layers = self.modelLayerNorm.expand_layer(self.layer_list)

        self.sample          = Sampling("Sampling", self.device).last_only()
        self.sample_layers = self.sample.expand_layer(self.layer_list)

        self.global_output   = GlobalOutput("GlobalOutput", self.device).last_only()
        self.global_output_layers = self.global_output.expand_layer(self.layer_list)

        self.copy_embedding = Copy("CopyEmbedding", self.device, num_inputs=2, num_outputs=2)

        self.copy_o = Copy("CopyO", self.device, num_inputs=1, num_outputs=2)

        self.copy_d = Copy("CopyD", self.device, num_inputs=1, num_outputs=2)

        self.redist_p = Redist("RedistPartition", self.device, num_inputs=1, num_outputs=2)

        self.redist_a = Redist("RedistAggregation", self.device, num_inputs=2, num_outputs=1)

        # Save operations in an instance variable
        self.original_model_operations: list[Operations] = [
            self.global_input, self.gen_embedding, self.layerNormAttn, self.kqv, self.ropeAppend,
            self.decAttn, self.pfAttn, self.o, self.layerNormFFN, self.ug, self.activation, self.d,
            self.modelLayerNorm, self.getLogits, self.sample, self.global_output
        ]
        self.original_virtual_operations: list[Operations] = [self.copy_embedding, self.copy_o, self.copy_d, self.redist_p, self.redist_a]

        self.model_operations = self.original_model_operations
        self.virtual_operations = self.original_virtual_operations
        self.all_operations = self.model_operations + self.virtual_operations # NOTE(Ziren): for further nanosplit or auto search, which should keep the original operations since we need to change the strategy of optimization in the runtime.

        self.all_layer_operations: list[Operation_Layer] = []
        for operation in self.model_operations:
            self.all_layer_operations.extend(operation.children)

    def init_dependency(self):
        # ">>"是自定义的“连接符”，表示数据流动方向，用于计算图构建
        self.global_input.outputs["tokens"] >> self.gen_embedding.inputs["token"]

        self.gen_embedding.outputs["output"] >> self.copy_embedding.inputs["input_0"]
        # LayerNormAttn 的输入来自 copy_embedding
        self.copy_embedding.outputs["output_0"] >> self.layerNormAttn.inputs["input"]
        self.copy_embedding.outputs["output_1"] >> self.o.inputs["C"]

        # LayerNormAttn 的输出去往 KQV
        self.layerNormAttn.outputs["output"] >> self.kqv.inputs["A"]

        self.kqv.outputs["D"] >> self.ropeAppend.inputs["kqv"]

        self.ropeAppend.outputs["q"] >> self.redist_p.inputs["input_0"]
        self.redist_p.outputs["output_0"] >> self.decAttn.inputs["Q"]
        self.redist_p.outputs["output_1"] >> self.pfAttn.inputs["Q"]

        self.decAttn.outputs["output"] >> self.redist_a.inputs["input_0"]
        self.pfAttn.outputs["output"] >> self.redist_a.inputs["input_1"]
        self.redist_a.outputs["output_0"] >> self.o.inputs["A"]

        self.o.outputs["D"] >> self.copy_o.inputs["input_0"]
        self.copy_o.outputs["output_0"] >> self.layerNormFFN.inputs["input"]
        self.copy_o.outputs["output_1"] >> self.d.inputs["C"]

        self.layerNormFFN.outputs["output"] >> self.ug.inputs["A"]

        self.ug.outputs["D"] >> self.activation.inputs["input"]

        self.activation.outputs["output"] >> self.d.inputs["A"]


        self.d.outputs["D"] >> self.copy_d.inputs["input_0"]
        self.copy_d.outputs["output_0"] >> (self.copy_embedding.inputs["input_1"], True)
        self.copy_d.outputs["output_1"] >> self.modelLayerNorm.inputs["input"]

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]

        for operation in self.all_operations:
            operation.checkConnection()
    
    
    def init_executor(self):
        self.executor = Executor(self.all_layer_operations, self.layer_list)
        self.executor.plan_layer_ordering()

    def init_set_shape(self):
        self.global_input.setShape()
        self.gen_embedding.setShape(self.hidden_dim, self.vocab_size)
        self.layerNormAttn.setShape(self.hidden_dim)
        self.kqv.setShape(self.kqv_heads * self.head_dim, self.hidden_dim).setParameter(1.0, 0.0)
        #self.decAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        #self.pfAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        #self.ropeAppend.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.decAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=1)   # tp_size是tensor parallel size，不等于1时用于多gpu场景
        self.pfAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=1)
        self.ropeAppend.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=1)
        
        self.o.setShape(self.hidden_dim, self.hidden_dim).setParameter(1.0, 1.0)
        self.layerNormFFN.setShape(self.hidden_dim)
        self.ug.setShape(self.intermediate_dim * 2, self.hidden_dim).setParameter(1.0, 0.0)
        self.d.setShape(self.hidden_dim, self.intermediate_dim).setParameter(1.0, 1.0)
        self.activation.setShape(self.intermediate_dim)
        self.modelLayerNorm.setShape(self.hidden_dim)
        self.getLogits.setShape(self.vocab_size, self.hidden_dim).setParameter(1.0, 0.0)
        self.sample.setShape(self.vocab_size)
        self.global_output.setShape()
    
    def init_cached_weight(self, weight_path):
        self.kv_cache = KVCacheNone()
        self.init_operations()
        self.init_set_shape()
        self.init_set_weight(weight_path, False)

    def init_set_weight(self, weight_path, cached):
        weight_manager = WeightManager(self.pipeline_name, weight_path, cached, self.device)
        weight_manager.set_weight(self.model_operations, self.device)


    def init_category(self):
        # set category for loop operations
        self.layerNormAttn.set_category(CategoryType.COMP)
        self.kqv.set_category(CategoryType.COMP)
        self.ropeAppend.set_category(CategoryType.COMP)
        self.decAttn.set_category(CategoryType.MEM)
        self.pfAttn.set_category(CategoryType.COMP)
        self.layerNormFFN.set_category(CategoryType.COMP)
        self.o.set_category(CategoryType.COMP)
        self.ug.set_category(CategoryType.COMP)
        self.activation.set_category(CategoryType.COMP)
        self.d.set_category(CategoryType.COMP)

    def clear_batch_size(self):
        # init the batchsize to None
        for op in self.all_operations:
            op.setBatchSize(None)

    def config_batch_size(self):
        self.global_input.setBatchSize(self.global_batch_size)
        self.decAttn.setBatchSize(self.decode_batch_size)

    def config_algorithm(self):
        params = {
            "use_cuda_graph": self.is_cuda_graph_enabled,
        }

        self.gen_embedding.config_tag("cuda", params)

        if self.is_auto_search_enabled:
            for op in self.model_operations:
                print(f"op.name: {op.name}, op.original_name: {op.original_name}")
                if op.original_name in self.profile_result["operations"]:
                    algo_tag = self.profile_result["operations"][op.original_name][op.name]["algo_tag"]
                    op.config_tag(algo_tag, params)
        else:
            self.layerNormAttn.config_tag("cuda", params)
            self.kqv.config_tag("torch", params)
            self.ropeAppend.config_tag("cuda", params)
            self.decAttn.config_tag("batched_cuda", params)
            self.pfAttn.config_tag("batched_cuda", params)
            self.layerNormFFN.config_tag("cuda", params)
            self.o.config_tag("torch", params)
            self.ug.config_tag("torch", params)
            self.activation.config_tag("cuda", params)
            self.d.config_tag("torch", params)

        self.getLogits.config_tag("torch", params)
        self.modelLayerNorm.config_tag("cuda", params)
        self.sample.config_tag("cuda", params)

    '''def config_algorithm(self):
        params = {
            "use_cuda_graph": self.is_cuda_graph_enabled,
        }

        if 'torch' in self.gen_embedding.impl_map:
            impl_tag = 'torch'
        else:
            impl_tag = list(self.gen_embedding.impl_map.keys())[0]
        print(f"Using implementation: {impl_tag}")
        # self.gen_embedding.config_tag(impl_tag, params)
        self.gen_embedding.config_tag("cuda", params)
        
        if self.is_auto_search_enabled:
            for op in self.model_operations:
                print(f"op.name: {op.name}, op.original_name: {op.original_name}")
                if op.original_name in self.profile_result["operations"]:
                    algo_tag = self.profile_result["operations"][op.original_name][op.name]["algo_tag"]
                    op.config_tag(algo_tag, params)
        else:
            self.layerNormAttn.config_tag("cuda", params)
            self.kqv.config_tag("torch", params)
            self.ropeAppend.config_tag("cuda", params)
            self.decAttn.config_tag("batched_cuda", params)
            self.pfAttn.config_tag("batched_cuda", params)
            self.layerNormFFN.config_tag("cuda", params)
            self.o.config_tag("torch", params)
            self.ug.config_tag("torch", params)
            self.activation.config_tag("cuda", params)
            self.d.config_tag("torch", params)

        self.getLogits.config_tag("torch", params)
        self.modelLayerNorm.config_tag("torch", params)
        self.sample.config_tag("torch", params)'''


    def config_streams(self):
        self.global_input.set_stream((self.main_stream, self.total_sm))
        self.gen_embedding.set_stream((self.main_stream, self.total_sm))

        if self.is_auto_search_enabled:
            for op in self.model_operations:
                print(f"op.name: {op.name}, op.original_name: {op.original_name}")
                if op.original_name in self.profile_result["operations"]:
                    sm_count = self.profile_result["operations"][op.original_name][op.name]["p_value"]
                    op.set_stream(self.streams[op.category][sm_count])
        else:
            self.layerNormAttn.set_stream((self.main_stream, self.total_sm))
            self.kqv.set_stream((self.main_stream, self.total_sm))
            self.ropeAppend.set_stream((self.main_stream, self.total_sm))
            self.decAttn.set_stream((self.main_stream, self.total_sm))
            self.pfAttn.set_stream((self.main_stream, self.total_sm))
            self.layerNormFFN.set_stream((self.main_stream, self.total_sm))
            self.o.set_stream((self.main_stream, self.total_sm))
            self.ug.set_stream((self.main_stream, self.total_sm))
            self.activation.set_stream((self.main_stream, self.total_sm))
            self.d.set_stream((self.main_stream, self.total_sm))

        self.getLogits.set_stream((self.main_stream, self.total_sm))
        self.modelLayerNorm.set_stream((self.main_stream, self.total_sm))
        self.sample.set_stream((self.main_stream, self.total_sm))
        self.global_output.set_stream((self.main_stream, self.total_sm))

    def profile_config_streams(self, stream_tuple):
        for operation in self.model_operations:
            operation.set_stream(stream_tuple)

    def nanobatch_split(self):
        op_nanobatch_info_map: dict[str, tuple[NanoOpInfo, ...]] = {}       #NanoOpInfo类 记录每个 nano 批次的索引和大小
        extra_links: dict[str, list[tuple[str, bool]]] = {}
        if self.is_auto_search_enabled:
            operations = self.profile_result["operations"]
            for (op_basename, op_info) in operations.items():
                split_info_list = []
                for nano_op_name, nano_op_info in op_info.items():
                    split_info_list.append(NanoOpInfo(
                        batch_idx=nano_op_info["batch_idx"],
                        batch_size=nano_op_info["batch_size"]
                    ))
                    extra_links[nano_op_name] = nano_op_info["extra_dep"]   #extra_links：记录每个 nano 批次的额外依赖（如 LayerNormAttn_0 依赖 KQV_0）

                op_nanobatch_info_map[op_basename] = tuple(split_info_list) #op_nanobatch_info_map：记录每个算子的所有 nano 批次信息（每个批次的 idx 和 size）
        else:   #如果没有profile data，默认分成两批，通过 decode_batch_size 和 global - decode 区分
            info = (
                NanoOpInfo(
                    batch_idx=0,
                    batch_size=self.decode_batch_size
                ),  
                NanoOpInfo(
                    batch_idx=1,
                    batch_size=self.global_batch_size - self.decode_batch_size
                )
            )
            op_nanobatch_info_map = {
                "LayerNormAttn": copy.deepcopy(info),
                "KQV": copy.deepcopy(info),
                "RopeAppend": copy.deepcopy(info),
                "O": copy.deepcopy(info),
                "LayerNormFFN": copy.deepcopy(info),
                "UG": copy.deepcopy(info),
                "Activation": copy.deepcopy(info),
                "D": copy.deepcopy(info),
            }
            extra_links = {}    #这种情况下没有额外依赖

        print("op_nanobatch_info_map", op_nanobatch_info_map)
        print("extra_links", extra_links)

        model_ops, addtional_virtual_ops = split_nanobatch(self.original_model_operations, op_nanobatch_info_map, extra_links)  #调用 split_nanobatch 进行实际拆分
        # split_nanobatch 会根据分批信息和依赖，把原始算子列表拆分为多个 nano 批次算子，并处理依赖关系。

        self.model_operations = model_ops   # 更新Pipeline的算子列表，是后续执行/调度的实际算子对象
        self.all_operations = []
        self.all_layer_operations = []
        for op in model_ops + self.virtual_operations + addtional_virtual_ops:
            print("op.name", op.name, op.batch_size)
            self.all_operations.append(op)      # 包含所有算子，包括实际算子、原本的虚拟算子和新增的虚拟算子
        for operation in model_ops:
            self.all_layer_operations.extend(operation.children)    # 把每个算子的 `children`（即每一层的 Operation_Layer 对象）都加进来。

    def update_allocate_buffers(self):
        # Build list of buffers(op_device)
        buffers_list = []
        for operation in self.all_operations:
            for _, wrapper in operation.inputs.items():
                buffers_list.append(wrapper)
            for _, wrapper in operation.outputs.items():
                buffers_list.append(wrapper)

        # Allocate buffers for each devices seperatly
        bufferAllocator = BufferAllocator(buffers_list)
        bufferAllocator.create_dependency_graph()
        bufferAllocator.set_all_batchsize_by_linear_programming()
        
        bufferAllocator.allocate_buffer(self.device)
        print(f"Total allocated: {bufferAllocator.total_allocated / 1024 / 1024} MB in {self.device}")

    def update(self, new_input_infos, decode_batch_size = 0, is_profile = False, stream_name: str = "TEST_TOTAL", profile_result_path: Optional[str] = None, use_cuda_graph: bool = False, use_nano_split: bool = False):
        #print(f"\nPipeline update:")
        #print(f"- Stream name: {stream_name}")
        #print(f"- Is profile: {is_profile}")
        #print(f"- Decode batch size: {decode_batch_size}")
        if stream_name in self.profile_streams:     # 根据 stream_name 选择当前 CUDA stream 和 SM 数量，便于多流/多SM测试。
            stream, sm_count = self.profile_streams[stream_name]
            #print(f"- Stream SM count: {sm_count}")

        # 检查 cumsum_input 的状态
        '''if hasattr(self, 'cumsum_input'):
            print(f"- Cumsum input type: {type(self.cumsum_input)}")
            if isinstance(self.cumsum_input, torch.Tensor):
                print(f"- Cumsum input device: {self.cumsum_input.device}")
                print(f"- Cumsum input shape: {self.cumsum_input.shape}")
        else:
            print("- No cumsum_input found")'''
        # preprocess new_input_infos
        with prof_marker("update_step_0"):
            self.input_req_idx = []
            self.input_ids = []
            for item in new_input_infos:       # new_input_infos: list of (request_idx, input_ids)
                # print("item", item)
                self.input_req_idx.append(item[0])
                self.input_ids.append(item[1])
        with prof_marker("update_step_1"):
            # concatenate input_ids into a single tensor
            # 将输入数据拆分为 request 索引和 token id 列表，展平成一维，生成输入张量
            flattened = [item for sublist in self.input_ids for item in sublist]
            global_batch_size = len(flattened)
        with prof_marker("update_step_3"):
            # torch.tensor(flattened, ...) 构造一个一维的输入张量，方便批量处理和拷贝到模型的输入 buffer
            input_tensor = torch.tensor(flattened, dtype=torch.int32, device=self.device)

        # some assertions and configuration settings
        # 判断是否需要重新分配 buffer， 如果 batch size 发生变化，需要重新分配 buffer 和重建算子结构。
        if global_batch_size != self.global_batch_size or decode_batch_size != self.decode_batch_size:
            self.buffer_fixed = False
        else:
            self.buffer_fixed = True

        # 配置 CUDA Graph 相关参数，支持 CUDA Graph 加速，相关参数在此配置。
        self.plan_cuda_graph = False
        if use_cuda_graph and self.is_cuda_graph_enabled:
            assert decode_batch_size == self.decode_batch_size and global_batch_size == self.global_batch_size, "decode_batch_size and global_batch_size must be the same when use_cuda_graph is True"
        elif use_cuda_graph and not self.is_cuda_graph_enabled:
            self.plan_cuda_graph = True
        else:
            self.is_cuda_graph_enabled = False
        self.is_cuda_graph_enabled = use_cuda_graph

        # 加载 profile_data，如果有 profile_data 路径，则加载 auto-search 结果，后续用于算子拆分和流配置
        if profile_result_path is not None:
            self.is_auto_search_enabled = True
            with open(profile_result_path, "r") as f:
                self.profile_result = json.load(f)
        else:
            self.is_auto_search_enabled = False
            self.profile_result = None

        # update if batch size or decode batch size has changed
        # 如果有变动（buffer_fixed=T，在上面有判断过），则重新分配 buffer 和拆分算子
        if not self.buffer_fixed:
            with prof_marker("update_step_2"):
                self.global_batch_size = global_batch_size
                self.decode_batch_size = decode_batch_size  # 更新 batch size
                # print(f"batch_size: {self.batch_size}")
                # print("decode_batchsize: ", decode_batchsize)
                self.clear_batch_size()
                self.config_batch_size()    # 清空并重新配置所有算子的 batch size
                if use_nano_split:
                    self.nanobatch_split()      # 重新拆分算子
                self.update_allocate_buffers()  # 重新分配 buffer
                # print("finish update_allocate_buffers")
                # 配置流、算法、执行器：
                if is_profile:
                    self.profile_config_streams(self.profile_streams[stream_name])
                else:
                    self.config_streams()
                self.config_algorithm()
                self.init_executor()

            with prof_marker("update_step_4"):
                request_length = torch.tensor([len(x) for x in self.input_ids], dtype=torch.int32, device='cpu')
            with prof_marker("update_step_5"):
                self.cumsum_input = torch.cat([torch.tensor([0], dtype=torch.int32, device='cpu'), torch.cumsum(request_length, dim=0, dtype=torch.int32)]).tolist()

        # update in any cases
        with prof_marker("update_step_6"):
            self.kv_cache.update(self.cumsum_input, self.input_req_idx, decode_batch_size, use_cuda_graph= (not self.plan_cuda_graph) and self.is_cuda_graph_enabled)
        with prof_marker("update_step_7"):
            self.global_input.outputs["tokens"].tensor.copy_(input_tensor)
        with prof_marker("update_step_8"):
            self.ropeAppend.update(self.cumsum_input, decode_batch_size)
        with prof_marker("update_step_9"):
            self.decAttn.update(self.cumsum_input)
        with prof_marker("update_step_10"):
            self.pfAttn.update(self.cumsum_input)

    def run(self, file_name="./test_data/llama3-8B-flashinfer", filefolder_name="./test_data/llama3-8B-flashinfer_folder"):

        temp_out = torch.zeros(self.global_batch_size, dtype=torch.int32, device='cuda')

        # os.makedirs(f"./{filefolder_name}", exist_ok=True)

        self.executor.execute(temp_out, self.main_stream, plan_cuda_graph=self.plan_cuda_graph, is_cuda_graph_enabled= self.is_cuda_graph_enabled)
        # self.executor.print_debug(temp_out, file_name, filefolder_name=filefolder_name)

        with prof_marker("after_execute_before_return"):
            temp_out = temp_out.cpu()
        with prof_marker("after_execute_step_1"):
            new_tokens = [ [temp_out[idx-1].item()] for idx in self.cumsum_input[1:]]
        with prof_marker("after_execute_step_2"):
            output = []
        with prof_marker("after_execute_step_3"):
            for req_idx, new_token in zip(self.input_req_idx, new_tokens):
                # print(f"req_idx: {req_idx}, new_token: {new_token}")
                output.append((req_idx, new_token))
        return output
    
    # profile related functions
    def init_profile_data(self, append_mode=False):
        for operation in self.model_operations:
            operation.setup_profile(self.profile_dir, append_mode=append_mode, is_save_db=True)

    def profile_run(self):
        for operation in self.model_operations:
            if operation.batch_size > 0:
                with prof_marker(f"{operation.name}"):
                    print("Operation name:", operation.name)
                    operation.profile_all()

    def profile_print(self):
        for operation in self.model_operations:
            operation.print_profile()