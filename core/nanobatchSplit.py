from operations.operation_base import NanoOpInfo, Operations
from operations.virtualOp.virtual_ops import Redist

def split_nanobatch(op_list: list[Operations], op_nano_info_map: dict[str, tuple[NanoOpInfo, ...]], extra_links):
    nano_op_list: list[Operations] = []
    additional_virtual_ops = []
    device = op_list[0].device
    for op in op_list:
        if (op.name not in op_nano_info_map):#op_nanobatch_info_map记录每个op的所有 nano 批次信息
            nano_op_list.append(op)          #如果算子没有分批信息，直接加入 nano_op_list。
            continue
        elif len(op_nano_info_map[op.name]) == 1:  
            #如果只分成一个批次，直接设置 batch_size 后加入 nano_op_list。
            op.setBatchSize(op_nano_info_map[op.name][0].batch_size)
            nano_op_list.append(op)
            continue

        # ...否则需要拆分 ↓

        nano_op_info_list = list(op_nano_info_map[op.name])
        op.isNanoSplit = True
        op.nano_ops = []

        redists_in = []
        redists_out = []
        # 为需要拆分的算子插入 Redist 虚拟算子
        # 每个输入插入一个 Redist 虚拟算子，实现“1输入分流到N个批次”
        for key, value in op.inputs.items():
            op_redist = Redist(f"Nano_Dist_{op.name}_{key}", device, 1, len(nano_op_info_list))
            op_redist.clear_inputs_and_outputs_links()
            op_redist.set_input(value)      #将原算子的输入与虚拟 Redist 算子的输入端口进行连接和替换
            redists_in.append(op_redist)
            additional_virtual_ops.append(op_redist)
        
        #每个输出插入一个 Redist 虚拟算子，实现“N个批次输出合流到1输出”
        for key, value in op.outputs.items():
            # print(f"Nano_Dist_{op.name}_{key}")
            op_redist = Redist(f"Nano_Dist_{op.name}_{key}", device, len(nano_op_info_list), 1)
            op_redist.clear_inputs_and_outputs_links()
            op_redist.set_output(value)     #将原算子的输出与虚拟 Redist 算子的输出端口进行连接和替换
            redists_out.append(op_redist)
            additional_virtual_ops.append(op_redist)

        for info in nano_op_info_list:
            batch_idx = info.batch_idx
            copied_op = op.copy_nano(batch_idx)
            copied_op.setBatchSize(info.batch_size)
            # 将输入输出重分布器与对应的 Nano Batch相连
            for j, (key, value) in enumerate(copied_op.inputs.items()):
                redists_in[j].outputs[f"output_{batch_idx}"] >> value
            for j, (key, value) in enumerate(copied_op.outputs.items()):
                value >> redists_out[j].inputs[f"input_{batch_idx}"]
            
            nano_op_list.append(copied_op)          # 将拆分后的算子加入 nano_op_list，它是Operations对象的列表
    
    nano_op_map = {op.name : op for op in nano_op_list}     # 构建一个字典，方便通过算子名查找 nano_op_list 中的算子对象
    for key, list_of_values in extra_links.items():         # key: 当前nano算子名， list_of_values: [(依赖的算子名, 是否依赖前一层)]
        for value, depend_on_prev_layer in list_of_values:  # value: 被当前nano算子所依赖的算子名, depend_on_prev_layer: 是否依赖前一层
            op_key = nano_op_map.get(key)           # 获取 nano_op_list 中当前的算子对象
            op_value = nano_op_map.get(value)       # 获取 nano_op_list 中当前算子依赖的算子对象
            # print("key", key, "value", value)
            # find the name key and value in nano_op_list
            # print("op_key.name", op_key.name, "key", key)
            # print("op_value.name", op_value.name, "value", value)
            if op_key and op_value:                                             # 如果两个算子都找到了
                op_key.append_dependency((op_value, depend_on_prev_layer))      # 把依赖关系加到 op_key 的 extra_dep 列表里；该函数在operation_base.py中定义
            else:
                raise ValueError(f"Operation {key} or {value} not found in nano_op_list")

    return nano_op_list, additional_virtual_ops     # 返回拆分后的算子列表和新增的虚拟算子列表