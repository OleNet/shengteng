import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.core as core
import paddle.fluid as fluid
import numpy as np

registerd_op = {"elementwise_add": "AddParser",
                "matmul": "MatMulParser",
                "mul": "MulParser",
                "relu": "ReluParser",
                "softmax_with_cross_entropy": "SoftmaxWithCrossEntropyParser",
                "shape": "ShapeParser",
                "fill_constant": "FillConstantParser",
                "reduce_sum": "ReduceSumParser",
                # ernie 2020-12-28
                "elementwise_mul": "DotMulParser",
                "elementwise_div": "DotDivParser",
                "elementwise_pow": "DotPowParser",
                "pow": "PowParser",
                "gelu": "GeluParser",
                "sqrt": "SqrtParser",
                "log": "LogParser",
                "sum": "SumParser",
                "logical_not": "LogicalNotParser",
                "gather": "GatherParser",
                "scatter": "ScatterParser",
                "cast": "CastParser",
                "tanh": "TanhParser",
                "assign": "AssignParser",
                "softmax": "SoftMaxParser",
                "reshape2": "ReshapeParser",
                "transpose2": "TransposeParser",
                "layer_norm": "LayerNormParser",
                # ernie 2020-12-29
                "less_than": "LessParser",
                "mean": "MeanParser",
                "scale": "ScaleParser",
                # ernie 2020-12-30
                "slice": "SliceParser",
                #"top_k": "TopkParser",
                #"accuracy": "AccuracyParser",
                #"increment": "IncrementParser",
                "lookup_table": "LookupTableParser",
                "truncated_gaussian_random": "TruncatedNormalParser",

                "equal": "EqualParser",
                "expand": "ExpandParser",
                "range": "RangeParser",
                "squeeze2": "SqueezeParser",
                "unsqueeze2": "UnSqueezeParser",
                "uniform_random": "UniformRandomParser",


                ## backwords
                "matmul_grad": "MatMulGradParser",
                "mul_grad": "MulGradParser",
                "relu_grad": "ReluGradParser",
                "reduce_sum_grad": "ReduceSumGradParser",
                "softmax_with_cross_entropy_grad": "SoftmaxWithCrossEntropyGradParser",
                "sgd": "SGDParser",
                "adam": "AdamParser",

                "tanh_grad":"TanhGradParser",
                "log_grad":"LogGradParser",
                "pow_grad": "PowGradParser",
                "sqrt_grad": "SqrtGradParser",
                "gelu_grad": "GeluGradParser",
                "mean_grad": "MeanGradParser",
                
                "elementwise_mul_grad": "DotMulGradParser",
                "elementwise_add_grad": "DotAddGradParser",
                "elementwise_div_grad": "DotDivGradParser",

                "softmax_grad": "SoftmaxGradParser",
                #"slice_grad": "SliceGradParser",
                "reshape2_grad": "ReshapeGradParser",
                "gather_grad": "GatherGradParser",
                "transpose2_grad": "TransposeGradParser",
                "layer_norm_grad": "LayerNormGradParser",

                "expand_grad": "ExpandGradParser",

                "squeeze2_grad": "SqueezeParserGrad",
                "unsqueeze2_grad": "UnSqueezeParserGrad",


                }
global_cnt = -1
class AscendHelper(object):
    def __init__(self):
        self.dtype_map = {
            0: core.GEDataType.DT_BOOL,
            1: core.GEDataType.DT_INT16,
            2: core.GEDataType.DT_INT32,
            3: core.GEDataType.DT_INT64,
            4: core.GEDataType.DT_FLOAT16,
            5: core.GEDataType.DT_FLOAT,
            6: core.GEDataType.DT_DOUBLE
        }
    def dtype2ge(self, dtype):
        assert dtype in self.dtype_map, "dtype[%d] is not supported %d" % (dtype)
        return self.dtype_map[dtype]

class AscendParserFactory(object):
    def __init__(self, graph, var2geop):
        self.graph = graph
        self.var2geop = var2geop
        
    def create_parse(self, parser_class):
        try:
            parser = globals()[parser_class](self.graph, self.var2geop)
            return parser
        except:
            raise ValueError("parser class %s does not exist" % parser_class)

class AscendParserBase(object):
    def __init__(self, graph, var2geop):        
        self.graph = graph
        self.var2geop = var2geop
        self.op = None
        self.ascend_helper = AscendHelper()

    def get_ge_input(self, input_var_name):
        assert input_var_name in self.var2geop, "var %s not created before" % (input_var_name)
        return self.var2geop[input_var_name]

    def update_output(self, graph, geop_list):
        #if self.parser_name == "range":
        #    import pdb
        #    pdb.set_trace()
        print("********************", self.var2geop)
        if self.parser_name != "sgd" :
            assert self.op.output_arg_names[0] not in self.var2geop, "var %s has been another op's output" % (self.op.output_arg_names[0])
        
        if (self.parser_name == "matmul_grad" or self.parser_name == "mul_grad") and len(self.op.output_arg_names) == 1:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.var2geop[self.op.output_arg_names[0]] = geop_list[-2]
            return 
        
        if self.parser_name == "layer_norm" or self.parser_name == "layer_norm_grad":
            self.var2geop[self.op.output_arg_names[2]] = geop_list[-3]
        else:
            self.var2geop[self.op.output_arg_names[0]] = geop_list[-1]
        
        op_list = ["softmax_with_cross_entropy", "matmul_grad", "mul_grad", \
                   "top_k", "elementwise_mul_grad", "elementwise_add_grad",
                   "elementwise_div_grad", "reshape2", "transpose2"]
        #if self.parser_name == "softmax_with_cross_entropy" or self.parser_name == "matmul_grad" or self.parser_name == "mul_grad" or self.parser_name == "top_k":
        if self.parser_name in op_list:
            # hard code for test
            if len(self.op.output_arg_names) > 1:
                self.var2geop[self.op.output_arg_names[1]] = geop_list[-2]
       
        if self.parser_name == "layer_norm":
            self.var2geop[self.op.output_arg_names[1]] = geop_list[-1]
            self.var2geop[self.op.output_arg_names[0]] = geop_list[-2]
        
        if self.parser_name == "layer_norm_grad":
            self.var2geop[self.op.output_arg_names[1]] = geop_list[-2]
            self.var2geop[self.op.output_arg_names[0]] = geop_list[-1]

        op_list2 = ["top_k", "reshape2", "transpose2", "layer_norm", "layer_norm_grad"]
        if self.parser_name in op_list2:
            self.graph.add_op(geop_list[0])
        else:
            for geop in geop_list:
                self.graph.add_op(geop)
        # print("update_output for op", self.op.type)

    def apply(self, op):
        self.op = op
        assert self.op.type == self.parser_name, "op [%s] != parser_name[%s]" % (self.op.type, self.parser_name)
        print("begin to parse op %s" % (self.parser_name))
        geop_list = self._apply()
        self.update_output(self.graph, geop_list)
    
    def getid(self):
        global global_cnt
        global_cnt += 1
        return "." + str(global_cnt)

    def create_ge_tensor(self, shape, dtype, value):
        # just stub
        tensor_desc = core.GETensorDesc(core.GEShape(shape), core.GEFormat.FORMAT_ND, self.ascend_helper.dtype2ge(dtype))
        # tensor_desc = core.GETensorDesc(core.GEShape(shape), core.GEFormat.FORMAT_ND, core.GEDataType.DT_FLOAT)
        #c1_tensor_desc.set_real_dim_cnt(1)
        tensor = core.GETensor(tensor_desc)
        
        def dtype2np(index):
            if index == 5:
                return "float32"
            if index == 2:
                return "int32"
            print("dont support dtype")

        data = (value * np.ones((shape))).reshape(shape).astype(dtype2np(dtype)) #TODO paddle dtype to np type
        buf = data.tobytes()
        data_8 = np.frombuffer(buf, dtype=np.uint8)
        tensor.set_data(data_8)
        return tensor
    def create_shape_tensor(self):
        # just stub
        tensor_desc = core.GETensorDesc(core.GEShape([2]), core.GEFormat.FORMAT_ND, core.GEDataType.DT_INT32)
        #c1_tensor_desc.set_real_dim_cnt(1)
        tensor = core.GETensor(tensor_desc)

        data = np.ones((2)).astype("int32").reshape([2]) #TODO paddle dtype to np type
        data[0] = 64
        buf = data.tobytes()
        data_8 = np.frombuffer(buf, dtype=np.uint8)
        tensor.set_data(data_8)
        return tensor

### elementwise_op
class AddParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AddParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_add"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        add = core.GEOperatorFactory.create_operator("add" + self.getid(), "Add").set_input("x1", data_x1_shape).set_input("x2", data_x2_shape)
        return [add]

class DotMulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotMulParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_mul"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        mul = core.GEOperatorFactory.create_operator("dotmul" + self.getid(), "Mul").set_input("x1", data_x1_shape).set_input("x2", data_x2_shape)
        return [mul]

class DotDivParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotDivParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_div"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        div = core.GEOperatorFactory.create_operator("dotdiv" + self.getid(), "Div").set_input("x1", data_x1_shape).set_input("x2", data_x2_shape)
        return [div]

class DotPowParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotPowParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_pow"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        pow = core.GEOperatorFactory.create_operator("dotpow" + self.getid(), "Pow").set_input("x1", data_x1_shape).set_input("x2", data_x2_shape)
        return [pow]

class LessParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LessParser, self).__init__(graph, var2geop)
        self.parser_name = "less_than"

    def _apply(self):
        x1 = self.get_ge_input(self.op.input_arg_names[0])
        x2 = self.get_ge_input(self.op.input_arg_names[1])
        less_than = core.GEOperatorFactory.create_operator("less_than" + self.getid(), "Less").set_input("x1", x1).set_input("x2", x2)
        return [less_than]


## cal
class LogParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LogParser, self).__init__(graph, var2geop)
        self.parser_name = "log"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        log = core.GEOperatorFactory.create_operator("log" + self.getid(), "Log").set_input("x", data_x1_shape)
        return [log]

class SqrtParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SqrtParser, self).__init__(graph, var2geop)
        self.parser_name = "sqrt"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        sqrt = core.GEOperatorFactory.create_operator("sqrt" + self.getid(), "Sqrt").set_input("x", data_x1_shape)
        return [sqrt]

class PowParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(PowParser, self).__init__(graph, var2geop)
        self.parser_name = "pow"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[0])
        factor = self.op.attr("factor")
        print("factor: ", factor)
        pow_value = core.GEOperatorFactory.create_operator("pow" + self.getid(), "Power").set_input("x", x).set_attr_float("power", factor)#.set_attr_vec_float("scale", [1.0]).set_attr_vec_float("shift", [0.0])
        return [pow_value]

class SumParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SumParser, self).__init__(graph, var2geop)
        self.parser_name = "sum"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        len_list = len(self.op.input_arg_names)
        print("***********: ", len(self.op.input_arg_names))

        if len_list < 2:
            assert False, "the size of input list must large or equal 2"
        x1 = self.get_ge_input(self.op.input_arg_names[0])
        x2 = self.get_ge_input(self.op.input_arg_names[1])
        sum = core.GEOperatorFactory.create_operator("sum" + self.getid(), "Add").set_input("x1", x1).set_input("x2", x2)
        for i in range(2, len_list):
            x2 = self.get_ge_input(self.op.input_arg_names[2])
            sum = core.GEOperatorFactory.create_operator("sum" + self.getid(), "Add").set_input("x1", sum).set_input("x2", x2)

        return [sum]

class LogicalNotParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LogicalNotParser, self).__init__(graph, var2geop)
        self.parser_name = "logical_not"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0]) 
        logical_not = core.GEOperatorFactory.create_operator("logical_not" + self.getid(), "LogicalNot").set_input("x", data_x1_shape)
        return [logical_not]

class MeanParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MeanParser, self).__init__(graph, var2geop)
        self.parser_name = "mean"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[0])
        mean = core.GEOperatorFactory.create_operator("mean" + self.getid(), "ReduceMeanD").set_input("x", x).set_attr_bool("keep_dims", False).set_attr_vec_int32("axes", [])
        return [mean]

class ReduceSumParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceSumParser, self).__init__(graph, var2geop)
        self.parser_name = "reduce_sum"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("dim")
        keep_dims = self.op.attr("keep_dim")
        print("axes in reduce_sum: ", axes)
        print("keep_dims in reduce_sum: ", keep_dims)
        reduce_sum = core.GEOperatorFactory.create_operator("reduce_sum" + self.getid(), "ReduceSumD").set_input("x", x, 0).set_attr_vec_int32("axes", axes).set_attr_bool("keep_dims", keep_dims)
        return [reduce_sum]

#class IncrementParser(AscendParserBase):
#    def __init__(self, graph, var2geop):
#        super(IncrementParser, self).__init__(graph, var2geop)
#        self.parser_name = "increment"
#
#    def _apply(self): 
#        x = self.get_ge_input(self.op.input_arg_names[0])
#        step = self.op.attr("step") #self.get_ge_input(self.op.input_arg_names[1])
#        print("step: ", step)
#            
#        increment = core.GEOperatorFactory.create_operator("adds" + self.getid(), "Adds").set_input("x", x).set_attr_float("value", step) #set_input("x2", bias)
#        
#        return [increment]

## matrix cal
class MatMulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MatMulParser, self).__init__(graph, var2geop)
        self.parser_name = "matmul"

    def _apply(self):
        x1 = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        matmul = core.GEOperatorFactory.create_operator("matmul" + self.getid(), "MatMul").set_input("x1", x1).set_input("x2", data_x2_shape)
        return [matmul]

class MulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MulParser, self).__init__(graph, var2geop)
        self.parser_name = "mul"

    def _apply(self):
        print("======", self.op.input_arg_names)
        print("======", self.op.output_arg_names)
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        print(self.op.block.var(self.op.input_arg_names[0]).shape)
        print(self.op.block.var(self.op.input_arg_names[1]).shape)
        
        matmul = core.GEOperatorFactory.create_operator("mul" + self.getid(), "MatMul").set_input("x1", data_x1_shape).set_input("x2", data_x2_shape)
        print(self.getid())
        return [matmul]

class LayerNormParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LayerNormParser, self).__init__(graph, var2geop)
        self.parser_name = "layer_norm"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[2])
        scale = self.get_ge_input(self.op.input_arg_names[1])
        bias = self.get_ge_input(self.op.input_arg_names[0])  
        print("========", self.op.input_arg_names)
        print(self.op.output_arg_names)
        epsilon = self.op.attr("epsilon")
        begin_norm_axis = self.op.attr("begin_norm_axis")
        print("epsilon: ", epsilon)
        print("begin_norm_axis", begin_norm_axis)


        shape_tensor = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Shape").set_input("x", x)
        scale_expand = core.GEOperatorFactory.create_operator("broadcast_to_d" + self.getid(), "BroadcastTo").set_input("x", scale).set_input("shape", shape_tensor)
        bias_expand =  core.GEOperatorFactory.create_operator("broadcast_to_d" + self.getid(), "BroadcastTo").set_input("x", bias).set_input("shape", shape_tensor)
        self.var2geop["scale_expand"] = scale_expand
        self.var2geop["bias_expand"] = bias_expand

        layer_norm = core.GEOperatorFactory.create_operator("layer_norm" + self.getid(), "LayerNorm").set_input("x", x).set_input("gamma", scale_expand).set_input("beta", bias_expand).set_attr_int32("begin_norm_axis", begin_norm_axis).set_attr_int32("begin_params_axis", begin_norm_axis).set_attr_float("epsilon", epsilon)

        return [layer_norm, layer_norm, layer_norm]


## activate function
class ReluParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReluParser, self).__init__(graph, var2geop)
        self.parser_name = "relu"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        relu = core.GEOperatorFactory.create_operator("relu" + self.getid(), "Relu").set_input("x", data_x1_shape)
        return [relu]

class GeluParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(GeluParser, self).__init__(graph, var2geop)
        self.parser_name = "gelu"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        gelu = core.GEOperatorFactory.create_operator("gelu" + self.getid(), "Gelu").set_input("x", data_x1_shape)
        return [gelu]

class TanhParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TanhParser, self).__init__(graph, var2geop)
        self.parser_name = "tanh"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        tanh = core.GEOperatorFactory.create_operator("tanh" + self.getid(), "Tanh").set_input("x", data_x1_shape)
        return [tanh]


## loss function
class SoftmaxWithCrossEntropyParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxWithCrossEntropyParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_with_cross_entropy"

    def _apply(self):
        print("=========", self.op.input_arg_names)
        label = self.get_ge_input(self.op.input_arg_names[0])
        logits = self.get_ge_input(self.op.input_arg_names[1])
        print(self.op.block.var(self.op.input_arg_names[1]))

        cls_num = self.op.block.var(self.op.input_arg_names[1]).shape[1]
        softmax = core.GEOperatorFactory.create_operator("softmax" + self.getid(), "SoftmaxV2").set_input("x", logits)
        label = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", label).set_attr_int32("dst_type", 3)


        tensoron = self.create_ge_tensor([1], 5, 1)
        on = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensoron)
        tensoroff = self.create_ge_tensor([1], 5, 0)
        off = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensoroff)
        onehot = core.GEOperatorFactory.create_operator("onehot" + self.getid(), "OneHotD").set_input("x", label).set_input("on_value", on).set_input("off_value", off).set_attr_int32("depth", cls_num)
        squeeze = core.GEOperatorFactory.create_operator("mul" + self.getid(), "Squeeze").set_input("x", onehot)
        #cast_squeeze = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", squeeze).set_attr_int32("dst_type", 5)#self.ascend_helper.dtype2ge(dtype))
        
        loss = core.GEOperatorFactory.create_operator("loss" + self.getid(), "SoftmaxCrossEntropyWithLogits").set_input("features", logits).set_input("labels", squeeze)
        print(self.getid())
        return [label, softmax, loss]

class SoftMaxParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftMaxParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax"

    def _apply(self):
        logits = self.get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axis")

        softmax = core.GEOperatorFactory.create_operator("softmax" + self.getid(), "SoftmaxV2").set_input("x", logits).set_attr_vec_int32("axes", [axes])

        return [softmax]


## general 
class ShapeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ShapeParser, self).__init__(graph, var2geop)
        self.parser_name = "shape"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[0])
        shape = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Shape").set_input("x", x)
        return [shape]

class FillConstantParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(FillConstantParser, self).__init__(graph, var2geop)
        self.parser_name = "fill_constant"

    def _apply(self):
        shape = self.op.attr("shape")
        dtype = self.op.attr("dtype")
        value = self.op.attr("value")
        print("self.op.output_arg_names[0]", self.op.output_arg_names[0])
        #if self.op.output_arg_names[0]=="fill_constant_3.tmp_0" \
        #   or self.op.output_arg_names[0]=="fill_constant_1.tmp_0" \
        #   or self.op.output_arg_names[0]=="fill_constant_5.tmp_0":
        #    import pdb
        #    pdb.set_trace()
        if dtype in [1,2,3,4,5,6]:
            value = float(self.op.attr("str_value"))
        print("shape: ", shape)
        print("dtype: ", dtype)
        print("value: ", value)
        tensor = self.create_ge_tensor(shape, dtype, value)
        const = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor)
        if self.op.block.var(self.op.output('Out')[0]).persistable:
            print("%s is Persistable in fill_constant" % (self.op.output('Out')[0]))
            var = core.GEOperatorFactory.create_operator(self.op.output('Out')[0], "Variable")
            var.update_output_desc("y", core.GETensorDesc(core.GEShape(shape), core.GEFormat.FORMAT_ND, core.GEDataType.DT_FLOAT))
            assign = core.GEOperatorFactory.create_operator("assign" + self.getid(), "Assign").set_input("value", const).set_input("ref", var).set_attr_bool("use_locking", True)
            return [const]
        else:
            print("self.op.output('Out')[0]: %s is not persistable in fill_constant"% (self.op.output('Out')[0]))
        return [const]

class TruncatedNormalParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TruncatedNormalParser, self).__init__(graph, var2geop)
        self.parser_name = "truncated_gaussian_random"

    def get_ge_tensor(self, value_list, dtype):
        np_in = np.array(value_list).astype(dtype)
        data_buffer = np.frombuffer(np_in.tobytes(), dtype=np.uint8)
        if dtype == np.float32:
            ge_dtype = core.GEDataType.DT_FLOAT
        elif dtype == np.int32:
            ge_dtype = core.GEDataType.DT_INT32

        shape_tensor_desc = core.GETensorDesc(core.GEShape([len(value_list)]), core.GEFormat.FORMAT_ND, ge_dtype)
        shape_tensor_desc.set_real_dim_cnt(1)
        shape_tensor = core.GETensor(shape_tensor_desc, data_buffer)
        data1 = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", shape_tensor)
        shape_const_desc = core.GETensorDesc(core.GEShape([len(value_list)]), core.GEFormat.FORMAT_ND, ge_dtype)
        data1.update_output_desc("y", shape_const_desc)

        return data1

    def _apply(self):
        shape = self.op.attr("shape")
        dtype = self.op.attr("dtype")
        mean = self.op.attr("mean")
        std = self.op.attr("std")
        seed = self.op.attr("seed")
        print("shape: ", shape)
        print("dtype: ", dtype)
        print("mean: ", mean)
        print("std: ", std)
        print("min: ", mean - 2*std)
        print("max: ", mean + 2*std)
        print("seed: ", seed)
        tensor1 = self.create_ge_tensor([len(shape)], 2, shape)
        shape_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor1)
        
        tensor2 = self.create_ge_tensor([1], dtype, mean)
        mean_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor2)
        
        tensor3 = self.create_ge_tensor([1], dtype, std)
        std_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor3)
        
        tensor4 = self.create_ge_tensor([1], dtype, mean-2*std)
        min_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor4)

        tensor5 = self.create_ge_tensor([1], dtype, mean+2*std)
        max_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor5)
        
        append_var = [shape_tensor, mean_tensor, std_tensor, min_tensor, max_tensor]
        name = "truncated_normal_input_tensor"
        for index, var in enumerate(append_var):
            self.var2geop[name + "_" + str(self.getid())] = shape_tensor
 
        #shape_tensor = self.get_ge_tensor(shape, np.int32)
        #mean_tensor = self.get_ge_tensor([mean], np.float32)
        #std_tensor = self.get_ge_tensor([std], np.float32)
        #min_tensor = self.get_ge_tensor([mean-2*std], np.float32)
        #max_tensor = self.get_ge_tensor([mean+2*std], np.float32)

        truncated_normal = core.GEOperatorFactory.create_operator("truncated_normal" + self.getid(), "ParameterizedTruncatedNormal").set_input("shape", shape_tensor).set_input("means", mean_tensor).set_input("stdevs", std_tensor).set_input("min", min_tensor).set_input("max", max_tensor).set_attr_int32("seed", 0)
        
        ## wirte the output of truncatedNormal from startup_program to main_program
        if self.op.block.var(self.op.output('Out')[0]).persistable:
            print("%s is Persistable in truncated_normal" % (self.op.output('Out')[0]))
            #var = core.GEOperatorFactory.create_operator(self.op.output('Out')[0], "Variable").set_input("x", truncated_normal)
            var = core.GEOperatorFactory.create_operator(self.op.output('Out')[0], "Variable")
            var.update_output_desc("y", core.GETensorDesc(core.GEShape(shape), core.GEFormat.FORMAT_ND, core.GEDataType.DT_FLOAT))
            assign = core.GEOperatorFactory.create_operator("assign" + self.getid(), "Assign").set_input("value", truncated_normal).set_input("ref", var)
            return [shape_tensor, mean_tensor, std_tensor, min_tensor, max_tensor, truncated_normal]
        else:
            print("self.op.output('Out')[0] is not persistable in truncated_noraml")
        return [truncated_normal] #[assign]

class GatherParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(GatherParser, self).__init__(graph, var2geop)
        self.parser_name = "gather"

    def _apply(self):
        print("======", self.op.input_arg_names)
        index = self.get_ge_input(self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[1])
        #row = len(self.op.block.var(self.op.input_arg_names[0]).shape)
        clo = self.op.block.var(self.op.input_arg_names[1]).shape[-1]
        gather = core.GEOperatorFactory.create_operator("gather" + self.getid(), "Gather").set_input("x", x).set_input("indices", index).set_attr_bool("validate_indices", True)
        return [gather]

class ScatterParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ScatterParser, self).__init__(graph, var2geop)
        self.parser_name = "scatter"

    def _apply(self):
        print("=======", self.op.input_arg_names)
        index = self.get_ge_input(self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[1])
        updates = self.get_ge_input(self.op.input_arg_names[2])
        overwrite = self.op.attr("overwrite")
        print(self.op.block.var(self.op.input_arg_names[1]))
        print(self.op.block.var(self.op.input_arg_names[2]))
        print("overwrite: ", overwrite)

        #tensoron = self.create_ge_tensor([3] ,2, 1)
        #const = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensoron)
        x_var = core.GEOperatorFactory.create_operator("variable" + self.getid(), "Variable").set_input("x", x)
        index_var = core.GEOperatorFactory.create_operator("variable" + self.getid(), "Variable").set_input("x", index)
        updates_var = core.GEOperatorFactory.create_operator("variable" + self.getid(), "Variable").set_input("x", updates)
        #self.var2geop["x_var"] = x_var
        #self.var2geop["index_var"] = index_var
        #self.var2geop["updates_var"] = updates_var
         
        if not overwrite:
            scatter_value = core.GEOperatorFactory.create_operator("scatter" + self.getid(), "ScatterAdd").set_input("var", x_var).set_input("indices", index_var).set_input("updates", updatesi_var)
        else:
            scatter_value = core.GEOperatorFactory.create_operator("scatter" + self.getid(), "ScatterUpdate").set_input("var", x_var).set_input("indices", index_var).set_input("updates", updates_var)  
        return [x_var, index_var, updates_var, scatter_value]

class CastParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(CastParser, self).__init__(graph, var2geop)
        self.parser_name = "cast"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[0])
        dtype = self.op.attr("out_dtype")
        print("dtype in cast: ", dtype)
        cast = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", x).set_attr_int32("dst_type", dtype)#self.ascend_helper.dtype2ge(dtype))
        return [cast]

class AssignParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AssignParser, self).__init__(graph, var2geop)
        self.parser_name = "assign"

    def _apply(self):
        const = self.get_ge_input(self.op.input_arg_names[0])
        var = self.get_ge_input(self.op.input_arg_names[1])
        assign = core.GEOperatorFactory.create_operator("assign" + self.getid(), "Assign").set_input("value", const).set_input("ref", var)
        return [assign]

class EqualParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(EqualParser, self).__init__(graph, var2geop)
        self.parser_name = "equal"

    def _apply(self):
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        data_x2_shape = self.get_ge_input(self.op.input_arg_names[1])
        assign = core.GEOperatorFactory.create_operator("equal" + self.getid(), "Equal").set_input("x1", data_x1_shape).set_input("x2", data_x2_shape)
        return [assign]


class ExpandParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ExpandParser, self).__init__(graph, var2geop)
        self.parser_name = "expand"

    def _apply(self):
        #print('>>>>>>>>>', self.op.input_arg_names)
        #print('>>>>>>>self.op', self.op.attr('expand_times'))
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        expand_times = self.op.attr('expand_times')

        #e_times = core.GEOperatorFactory.create_operator("expand_times" + self.getid(), "Const").set_attr_tensor("value", expand_times)

        tensor = self.create_ge_tensor([len(expand_times)], 2, expand_times)
        #expand_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor)
        #assign = core.GEOperatorFactory.create_operator("tile" + self.getid(), "Tile").set_input("x", data_x1_shape).set_input("multiples", expand_tensor )
        expand_tensor = core.GEOperatorFactory.create_operator("const" +self.getid()+"9999999999999", "Const").set_attr_tensor("value", tensor)
        assign = core.GEOperatorFactory.create_operator("tile" + self.getid()+"9283474579", "Tile").set_input("x", data_x1_shape).set_input("multiples", expand_tensor )
        return [assign]


class ExpandGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ExpandGradParser, self).__init__(graph, var2geop)
        self.parser_name = "expand_grad"

    def _apply(self):
        out_grad = self.get_ge_input(self.op.input_arg_names[0])
        expand_times = self.op.attr('expand_times')
        
        tensor = self.create_ge_tensor([len(expand_times)], 2, expand_times)
        const_shape = core.GEOperatorFactory.create_operator("shape" + self.getid() + "98782919928", "Const").set_attr_tensor("value", tensor) 
        x_grad = core.GEOperatorFactory.create_operator("expand" + self.getid() + "8888888888888", "Tile").set_input("x", out_grad).set_input("multiples", const_shape)

        return [x_grad]


class RangeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(RangeParser, self).__init__(graph, var2geop)
        self.parser_name = "range"

    def _apply(self):
        # TODO not support range type yet
        """
        start = self.get_ge_input(self.op.input_arg_names[0])
        end = self.get_ge_input(self.op.input_arg_names[1])
        delta = self.get_ge_input(self.op.input_arg_names[2])
        """
        #start = self.op.attr("start")
        #end = self.op.attr("end")
        #delta = self.op.attr("delta")
        start = self.get_ge_input(self.op.input_arg_names[0])
        end = self.get_ge_input(self.op.input_arg_names[1])
        delta = self.get_ge_input(self.op.input_arg_names[2])
        print("~~~~~~~~~~~~~~~~~~~~~start: ", start)
        print("~~~~~~~~~~~~~~~~~~~~~start: ", end)
        print("~~~~~~~~~~~~~~~~~~~~~start: ", delta)
        #dtype = self.ascend_helper.dtype2ge(5)
        #dtype = self.op.attr("dtype")


        #const_bias = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor_bias)
        #tmp_start_tensor = self.create_ge_tensor([1], dtype, start)
        #start_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tmp_start_tensor)

        #tmp_end_tensor = self.create_ge_tensor([1], dtype, end)
        #end_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tmp_end_tensor)

        #delta_start_tensor = self.create_ge_tensor([1], dtype, delta)
        #delta_tensor = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tmp_delta_tensor)

        #append_var = [start_tensor, end_tensor, delta_tensor]
        #name = "range"
        #for index, var in enumerate(append_var):
        #    self.var2geop[name + "_" + str(self.getid())] = shape_tensor
        #shape_tensor = self.get_ge_tensor(shape, np.int32)
        #mean_tensor = self.get_ge_tensor([mean], np.float32)
        #std_tensor = self.get_ge_tensor([std], np.float32)
        #min_tensor = self.get_ge_tensor([mean-2*std], np.float32)
        #max_tensor = self.get_ge_tensor([mean+2*std], np.float32)


        ge_range = core.GEOperatorFactory.create_operator("range" + self.getid(), "Range")\
        .set_input("start", end)\
        .set_input("limit", start) \
        .set_input("delta", delta)
        # .update_output_desc("y", core.GETensorDesc(core.GEShape(shape), core.GEFormat.FORMAT_ND, core.GEDataType.DT_FLOAT))
        return [ge_range]


class SqueezeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SqueezeParser, self).__init__(graph, var2geop)
        self.parser_name = "squeeze2"

    def _apply(self):
        tensor = self.get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axes") 

        ge_range = core.GEOperatorFactory.create_operator("squeeze" + self.getid(), "Squeeze").set_input("x", tensor).set_attr_vec_int32("axes", axes)
        return [ge_range]


class UnSqueezeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(UnSqueezeParser, self).__init__(graph, var2geop)
        self.parser_name = "unsqueeze2"

    def _apply(self):
        tensor = self.get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axes") 

        ge_range = core.GEOperatorFactory.create_operator("unsqueeze" + self.getid(), "Unsqueeze").set_input("x", tensor).set_attr_vec_int32("axes", axes)
        return [ge_range]


class UniformRandomParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(UniformRandomParser, self).__init__(graph, var2geop)
        self.parser_name = "uniform_random"

    def _apply(self):
        shape = self.op.attr("shape")
        #dtype = self.op.attr("dtype")
        #min_v = self.op.attr("min")
        #max_v = self.op.attr("max")
        #seed = self.op.attr("seed")

        #grad = self.get_ge_input(self.op.input_arg_names[0])
        #x = self.get_ge_input(self.op.input_arg_names[1])
        #factor = self.op.attr("factor")


        #shape = self.op.attr("shape")
        axis = 0
        if shape[0] == -1:
            axis = 1
            shape = shape[1:]
        print("shape: ", shape)
        tensor = self.create_ge_tensor([len(shape)], 2, shape)
        shape_tensor = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Const").set_attr_tensor("value", tensor)

        """
        min_scale = self.create_ge_tensor([1], 5, min_v)
        min_scale = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", min_scale)
        min_tensor = core.GEOperatorFactory.create_operator("broadcast_to_d" + self.getid(), "BroadcastTo").set_input("x", min_scale).set_input("shape", shape_tensor)

        max_scale = self.create_ge_tensor([1], 5, max_v)
        max_scale = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", max_scale)
        max_tensor = core.GEOperatorFactory.create_operator("broadcast_to_d" + self.getid(), "BroadcastTo").set_input("x", max_scale).set_input("shape", shape_tensor)

        """

        ge_range = core.GEOperatorFactory.create_operator("uniform_random" + self.getid(), "RandomUniform")\
            .set_input("shape", shape_tensor)\
            .set_attr_int32("dtype", self.ascend_helper.dtype2ge(5))  \
            .set_attr_float("seed", 1)\
            .set_attr_float("seed2", 1) 
                        


        """
        (max_tensor - min_tensor) / 2 * (ge_range - (-1)) + min_tensor



        x_power = core.GEOperatorFactory.create_operator("x_power" + self.getid(), "Power").set_input("x", x).set_attr_float("power", factor - 1)
        x_power_mul_factor = core.GEOperatorFactory.create_operator("x_power_mul_factor" + self.getid(), "Mul").set_input("x1", x).set_input("x2", factor_tensor)
        x_power_mul_factor_grad = core.GEOperatorFactory.create_operator("x_power_mul_factor_grad" + self.getid(), "Mul").set_input("x1", x_power_mul_factor).set_input("x2", grad)
        return [shape_tensor, factor_scale, factor_tensor, x_power, x_power_mul_factor, x_power_mul_factor_grad]
        """

        return [ge_range]



class SqueezeGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SqueezeParser, self).__init__(graph, var2geop)
        self.parser_name = "squeeze2_grad"

    def _apply(self):
        tensor = self.get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axes") 

        ge_range = core.GEOperatorFactory.create_operator("squeeze2_grad" + self.getid(), "SqueezeGrad").set_input("x", tensor).set_attr_vec_int32("axes", axes)
        return [ge_range]


class UnSqueezeGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(UnSqueezeGradParser, self).__init__(graph, var2geop)
        self.parser_name = "unsqueeze2_grad"

    def _apply(self):
        tensor = self.get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axes") 

        ge_range = core.GEOperatorFactory.create_operator("unsqueeze2_grad" + self.getid(), "UnsqueezeGrad").set_input("x", tensor).set_attr_vec_int32("axes", axes)
        return [ge_range]


class ScaleParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ScaleParser, self).__init__(graph, var2geop)
        self.parser_name = "scale"

    def _apply(self): 
        x = self.get_ge_input(self.op.input_arg_names[0])
        scale = self.op.attr("scale") #self.get_ge_input(self.op.input_arg_names[1])
        bias = self.op.attr("bias")
        bias_after_scale = self.op.attr("bias_after_scale")
        print("scale: ", scale)
        print("bias: ", bias)
        print("bias_after_scale: ", bias_after_scale)
        if bias_after_scale:
            scale_value = core.GEOperatorFactory.create_operator("scale" + self.getid(), "Power").set_input("x", x).set_attr_float("power", 1.0).set_attr_float("scale", scale).set_attr_float("shift", bias)
        else:
            x_add_bias = core.GEOperatorFactory.create_operator("adds" + self.getid(), "Adds").set_input("x", x).set_attr_float("value", bias) #set_input("x2", bias)
            scale_value = core.GEOperatorFactory.create_operator("scale" + self.getid(), "Power").set_input("x", x_add_bias).set_attr_float("power", 1.0).set_attr_float("scale", scale).set_attr_float("shift", 0.0) 
            #tensor_zeros = core.GEOperatorFactory.create_operator("zeroslike" + self.getid(), "ZerosLike").set_input("x", x)
            #bias_ = self.create_ge_tensor([1], 5, bias)     
            #const_bias = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor_bias)
        return [scale_value]

class SliceParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SliceParser, self).__init__(graph, var2geop)
        self.parser_name = "slice"

    def _apply(self):
        x = self.get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("axes") #self.get_ge_input(self.op.input_arg_names[1])
        starts = self.op.attr("starts")
        ends = self.op.attr("ends")
        print("org axes: ", axes, len(axes))
        print("org starts: ", starts, len(starts))
        print("org ends: ", ends, len(ends))
        
        x_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        len_shape = len(x_shape)
        axes_cor = list(range(len_shape))
        starts_cor, ends_cor = [], []
        cnt = 0
        for i in range(len_shape):
            starts_cor.append(starts[cnt] if i in axes else 0)
            if i in axes and ends[cnt] <= x_shape[i]:
                ends_cor.append(ends[cnt])
            else:
                ends_cor.append(x_shape[i])
            if i in axes:
                cnt += 1
            #print(starts_cor, ends_cor, cnt, i)
        size = [ends_cor[i] - starts_cor[i] for i in range(len(axes_cor))]
        print("size: ", size)
        print("cor axes: ", axes_cor)
        print("cor starts: ", starts_cor)
        print("cor ends: ", ends_cor)
        
        assert len(axes) == len(starts) == len(ends), "the three fields must have same size" 
        slice_value = core.GEOperatorFactory.create_operator("slice" + self.getid(), "SliceD").set_input("x", x).set_attr_vec_int32("offsets", starts_cor).set_attr_vec_int32("size", size)
        
        return [slice_value]

class ReshapeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReshapeParser, self).__init__(graph, var2geop)
        self.parser_name = "reshape2"

    def _apply(self):
        print("swbuf:", self.op.input_arg_names)
        shape = self.op.attr("shape")
        axis = 0
        if shape[0] == -1:
            axis = 1
            shape = shape[1:]
        print("shape: ", shape)
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        tensor = self.create_ge_tensor([len(shape)], 2, shape)
        const_shape = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Const").set_attr_tensor("value", tensor)
        reshape = core.GEOperatorFactory.create_operator("reshape" + self.getid(), "Reshape").set_input("x", data_x1_shape).set_input("shape", const_shape).set_attr_int32("axis", axis)
        
        return [reshape, reshape]

class TransposeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TransposeParser, self).__init__(graph, var2geop)
        self.parser_name = "transpose2"

    def _apply(self):
        print("swbuf:", self.op.input_arg_names)
        perm = self.op.attr("axis")
        print("perm: ", perm)
        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
        transpose = core.GEOperatorFactory.create_operator("transpose" + self.getid(), "TransposeD").set_input("x", data_x1_shape).set_attr_vec_int32("perm", perm)
        return [transpose, transpose]

#class AccuracyParser(AscendParserBase):
#    def __init__(self, graph, var2geop):
#        super(AccuracyParser, self).__init__(graph, var2geop)
#        self.parser_name = "accuracy"
#
#    def _apply(self):
#        print("=============", self.op.input_arg_names)
#        print(self.op.block.var(self.op.input_arg_names[2]).dtype)
#        logits = self.get_ge_input(self.op.input_arg_names[2])
#        pred = self.get_ge_input(self.op.input_arg_names[0])
#        label = self.get_ge_input(self.op.input_arg_names[1])
#        
#        ## acc
#        pred = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", pred, 1).set_attr_int32("dst_type", 2)
#        label = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", label, 0).set_attr_int32("dst_type", 2)
#        equal = core.GEOperatorFactory.create_operator("equal" + self.getid(), "Equal").set_input("x1", pred, 0).set_input("x2", label)
#        cast = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", equal, 0).set_attr_int32("dst_type", 2)
#        acc = core.GEOperatorFactory.create_operator("mean" + self.getid(), "ReduceMeanD").set_input("x", cast, 0).set_attr_bool("keep_dims", False).set_attr_vec_int32("axes", [])
#        ## total
#        tensor_ones = core.GEOperatorFactory.create_operator("oneslike" + self.getid(), "OnesLike").set_input("x", label, 0)
#        total = core.GEOperatorFactory.create_operator("reduce_sum" + self.getid(), "ReduceSumD").set_input("x", tensor_ones, 0).set_attr_vec_int32("axes", []).set_attr_bool("keep_dims", False)
#     
#        return [total, acc]
#
#class TopkParser(AscendParserBase):
#    def __init__(self, graph, var2geop):
#        super(TopkParser, self).__init__(graph, var2geop)
#        self.parser_name = "top_k"
#
#    def _apply(self):
#        data_x1_shape = self.get_ge_input(self.op.input_arg_names[0])
#        print("======", self.op.block.var(self.op.input_arg_names[0]).dtype)
#    
#        k = self.op.attr("k")
#        print("k: ", k)
#        tensor = self.create_ge_tensor([1], 2, k)
#        const_k = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensor)
#        k_cast = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", const_k).set_attr_int32("dst_type", 2)
#        #x = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", data_x1_shape).set_attr_int32("dst_type", 4)
#        topk = core.GEOperatorFactory.create_operator("topk" + self.getid(), "TopK").set_input("x", data_x1_shape).set_input("k", k_cast)
#        #value = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", topk, 0).set_attr_int32("dst_type", 5)
#        #index = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", topk, 1).set_attr_int32("dst_type", 3)
#        return [topk, topk]   # [value, index]






## grad
class ReduceSumGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceSumGradParser, self).__init__(graph, var2geop)
        self.parser_name = "reduce_sum_grad"

    def _apply(self):
        print("self.op.input_arg_names[0]: ", self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[0])
        input = self.get_ge_input(self.op.input_arg_names[1])

        # shape_tensor = self.create_shape_tensor()
        # # shape_tensor = self.create_ge_tensor([2], 2, 5)
        # const = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", shape_tensor)
        # self.var2geop["tmp_const"] = const
        shape_tensor = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Shape").set_input("x", input, 0)


        tensoron = self.create_ge_tensor([1], 2, -1)
        const = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensoron)
        self.var2geop["tmp_const"] = const

        # # unsqueeze = core.GEOperatorFactory.create_operator("Unsqueeze" + self.getid(), "Unsqueeze").set_input("x", shape_tensor).set_attr_vec_int32("axes", [1])
        # expand = core.GEOperatorFactory.create_operator("expand" + self.getid(), "ExpandDims").set_input("x", shape_tensor).set_input("axis", const)

        reduce_sum = core.GEOperatorFactory.create_operator("broadcast_to_d" + self.getid(), "BroadcastTo").set_input("x", x).set_input("shape", shape_tensor)
        reduce_sum = core.GEOperatorFactory.create_operator("expand" + self.getid(), "ExpandDims").set_input("x", reduce_sum).set_input("axis", const)
        return [reduce_sum]

class MatMulGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MatMulGradParser, self).__init__(graph, var2geop)
        self.parser_name = "matmul_grad"

    def _apply(self):
        out_grad = self.get_ge_input(self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[1])
        y = self.get_ge_input(self.op.input_arg_names[2])
        print('>>>>>>>>>>>>', x)
        print('>>>>>>>>>>>>', y)

        x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "MatMul").set_input("x1", out_grad).set_input("x2", y).set_attr_bool("transpose_x1", False).set_attr_bool("transpose_x2", True)
        y_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "MatMul").set_input("x1", x).set_input("x2", out_grad).set_attr_bool("transpose_x1", True).set_attr_bool("transpose_x2", False)
        return [y_grad, x_grad]

class MulGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MulGradParser, self).__init__(graph, var2geop)
        self.parser_name = "mul_grad"

    def _apply(self):
        out_grad = self.get_ge_input(self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[1])
        y = self.get_ge_input(self.op.input_arg_names[2])
        print("####################self.op.input_arg_names: ", self.op.input_arg_names)

        x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "MatMul").set_input("x1", out_grad).set_input("x2", y).set_attr_bool("transpose_x1", False).set_attr_bool("transpose_x2", True)
        y_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "MatMul").set_input("x1", x).set_input("x2", out_grad).set_attr_bool("transpose_x1", True).set_attr_bool("transpose_x2", False)
        print(self.getid())

        return [y_grad, x_grad]

class ReluGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReluGradParser, self).__init__(graph, var2geop)
        self.parser_name = "relu_grad"

    def _apply(self):
        out = self.get_ge_input(self.op.input_arg_names[0])
        out_grad = self.get_ge_input(self.op.input_arg_names[1])
        relu_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "ReluGrad").set_input("gradients", out_grad).set_input("features", out)
        return [relu_grad]

#class GeluGradParser(AscendParserBase):
#    def __init__(self, graph, var2geop):
#        super(GeluGradParser, self).__init__(graph, var2geop)
#        self.parser_name = "gelu_grad"
#
#    def _apply(self):
#        out = self.get_ge_input(self.op.input_arg_names[0])
#        out_grad = self.get_ge_input(self.op.input_arg_names[1])
#        gelu_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "GeluGrad").set_input("dy", out_grad).set_input("x", out).set_input("y", )
#        return [gelu_grad]

class SoftmaxWithCrossEntropyGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxWithCrossEntropyGradParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_with_cross_entropy_grad"

    def _apply(self):
        label = self.get_ge_input(self.op.input_arg_names[0])
        loss_grad = self.get_ge_input(self.op.input_arg_names[1])
        softmax = self.get_ge_input(self.op.input_arg_names[2])
        cls_num = self.op.block.var(self.op.input_arg_names[2]).shape[1]

        tensoron = self.create_ge_tensor([1], 5, 1)
        on = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensoron)
        tensoroff = self.create_ge_tensor([1], 5, 0)
        off = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensoroff)
        label = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", label).set_attr_int32("dst_type", 3)
        onehot = core.GEOperatorFactory.create_operator("onehot" + self.getid(), "OneHotD").set_input("x", label).set_input("on_value", on).set_input("off_value", off).set_attr_int32("depth", cls_num)
        # the fuck onehot will add a demension, so must call squeeze afterward
        squeeze = core.GEOperatorFactory.create_operator("mul" + self.getid(), "Squeeze").set_input("x", onehot)
        sub = core.GEOperatorFactory.create_operator("sub" + self.getid(), "Sub").set_input("x1", softmax).set_input("x2", squeeze)
        #cast_sub = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", sub).set_attr_int32("dst_type", 5)#self.ascend_helper.dtype2ge(dtype))
        grad = core.GEOperatorFactory.create_operator("mul" + self.getid(), "Mul").set_input("x1", loss_grad).set_input("x2", sub)
        print(self.getid())
        return [on, off, label, onehot, grad]

#class SqrtGradParser(AscendParserBase):
#    def __init__(self, graph, var2geop):
#        super(SqrtGradParser, self).__init__(graph, var2geop)
#        self.parser_name = "sqrt_grad"
#
#    def _apply(self):
#        print("===============", self.op.input_arg_names)
#        out = self.get_ge_input(self.op.input_arg_names[0])
#        out_grad = self.get_ge_input(self.op.input_arg_names[1])
#        sqrt_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "SqrtGrad").set_input("y", out).set_input("dy", out_grad)
#        return [sqrt_grad]

class DotMulGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotMulGradParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_mul_grad"

    def _apply(self):
        print("==============", self.op.input_arg_names)
        out_grad = self.get_ge_input(self.op.input_arg_names[0])
        out_1 = self.get_ge_input(self.op.input_arg_names[1])
        out_2 = self.get_ge_input(self.op.input_arg_names[2])
        
        x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "Mul").set_input("x1", out_grad).set_input("x2", out_2)
        y_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "Mul").set_input("x1", out_1).set_input("x2", out_grad)

        return [y_grad, x_grad]

class DotAddGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotAddGradParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_add_grad"

    def _apply(self):
        ## 仅能用于x和y的纬度一致的情况
        print("==============", self.op.input_arg_names)
        out_grad = self.get_ge_input(self.op.input_arg_names[0])
        out_1 = self.get_ge_input(self.op.input_arg_names[1])
        out_2 = self.get_ge_input(self.op.input_arg_names[2])
        
        # x_grad
        x_grad = out_grad
        while len(self.op.block.var(self.op.input_arg_names[0]).shape) > len(self.op.block.var(self.op.input_arg_names[1]).shape):
             x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "ReduceSumD").set_input("x", x_grad).set_attr_vec_int32("axes", 0).set_attr_bool("keep_dims", False)
        for axis, size in enumerate(self.op.block.var(self.op.input_arg_names[1]).shape):
            if size == 1:
                x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "ReduceSumD").set_input("x", x_grad).set_attr_vec_int32("axes", axis).set_attr_bool("keep_dims", True)
        
        # y_grad
        y_grad = out_grad
        while len(self.op.block.var(self.op.input_arg_names[0]).shape) > len(self.op.block.var(self.op.input_arg_names[2]).shape):
             y_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "ReduceSumD").set_input("x", y_grad).set_attr_vec_int32("axes", 0).set_attr_bool("keep_dims", False)
        for axis, size in enumerate(self.op.block.var(self.op.input_arg_names[2]).shape):
            if size == 1:
                y_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "ReduceSumD").set_input("x", y_grad).set_attr_vec_int32("axes", axis).set_attr_bool("keep_dims", True)
        
        return [y_grad, x_grad]

class DotDivGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(DotDivGradParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_div_grad"

    def _apply(self):
        print("==============", self.op.input_arg_names)
        out = self.get_ge_input(self.op.input_arg_names[0])
        out_grad = self.get_ge_input(self.op.input_arg_names[1])
        x = self.get_ge_input(self.op.input_arg_names[2])
        y = self.get_ge_input(self.op.input_arg_names[3])
        x_dtype = self.op.block.var(self.op.input_arg_names[2]).dtype
       
        #tensoron = self.create_ge_tensor([1], 5, 1)
        #on = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", tensoron)
        
        # x_grad
        tensor_zeros = core.GEOperatorFactory.create_operator("zeroslike" + self.getid(), "ZerosLike").set_input("x", x)
        x_zero = core.GEOperatorFactory.create_operator("equal" + self.getid(), "Equal").set_input("x1", x).set_input("x2", tensor_zeros)
        x_nozero = core.GEOperatorFactory.create_operator("logical_not" + self.getid(), "LogicalNot").set_input("x", x_zero)
        x_nozero_f = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", x_nozero).set_attr_int32("dst_type", 3)
        x_grad_w = core.GEOperatorFactory.create_operator("dotdiv" + self.getid(), "Div").set_input("x1", x_nozero_f).set_input("x2", y)
        x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "Mul").set_input("x1", x_grad_w).set_input("x2", out_grad)

        # y_grad
        neg_out = core.GEOperatorFactory.create_operator("neg" + self.getid(), "Neg").set_input("x", out)
        y_grad_w = core.GEOperatorFactory.create_operator("dotdiv" + self.getid(), "Div").set_input("x1", neg_out).set_input("x2", y)
        y_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "Mul").set_input("x1", y_grad_w).set_input("x2", out_grad)

        #x_grad = out_grad
        #y_grad = out_grad

        return [tensor_zeros, x_zero, x_nozero, x_nozero_f, x_grad_w, neg_out, y_grad_w, y_grad, x_grad]

class SoftmaxGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxGradParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_grad"

    def _apply(self):
        print("===========", self,op.input_arg_names)
        out = self.get_ge_input(self.op.input_arg_names[0])
        out_grad = self.get_ge_input(self.op.input_arg_names[1])

        x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "SoftmaxGrad").set_input("softmax", out).set_input("softmax_grad", out_grad)

        return [x_grad]

class SliceGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SliceGradParser, self).__init__(graph, var2geop)
        self.parser_name = "slice_grad"

    def _apply(self):
        print("===========", self,op.input_arg_names)
        out = self.get_ge_input(self.op.input_arg_names[0])
        out_grad = self.get_ge_input(self.op.input_arg_names[1])

        x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "SoftmaxGrad").set_input("softmax", out).set_input("softmax_grad", out_grad)

        return [x_grad]

class ReshapeGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReshapeGradParser, self).__init__(graph, var2geop)
        self.parser_name = "reshape2_grad"

    def _apply(self):
        print("===========", self.op.input_arg_names)
        out_grad = self.get_ge_input(self.op.input_arg_names[0])
        x_shape = self.get_ge_input(self.op.input_arg_names[1])
        x_shape_list = self.op.block.var(self.op.input_arg_names[1]).shape
        x_shape_delzero = x_shape_list[1:]
        
        tensor = self.create_ge_tensor([len(x_shape_delzero)], 2, x_shape_delzero)
        const_shape = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Const").set_attr_tensor("value", tensor) 
        x_grad = core.GEOperatorFactory.create_operator("reshape" + self.getid(), "Reshape").set_input("x", out_grad).set_input("shape", const_shape)#.set_attr_int32("axis", axis)

        return [x_grad]

class GatherGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(GatherGradParser, self).__init__(graph, var2geop)
        self.parser_name = "gather_grad"
    
    def get_variable(self, shape, dtype, tensor):
        if dtype == "int32":
            type = core.GEDataType.DT_INT32
        elif dtype == "float32":
            type = core.GEDataType.DT_FLOAT

        var = core.GEOperatorFactory.create_operator("variable" + self.getid(), "Variable")
        var.update_output_desc("y", core.GETensorDesc(core.GEShape(shape), core.GEFormat.FORMAT_ND, type))
        assign = core.GEOperatorFactory.create_operator("assign" + self.getid(), "Assign").set_input("value", tensor).set_input("ref", var)
        
        return assign

    def _apply(self):
        print("===========", self.op.input_arg_names)
        index = self.get_ge_input(self.op.input_arg_names[0])
        out_grad = self.get_ge_input(self.op.input_arg_names[1])
        x = self.get_ge_input(self.op.input_arg_names[2])
        index_shape = self.op.block.var(self.op.input_arg_names[0]).shape
        out_grad_shape = self.op.block.var(self.op.input_arg_names[1]).shape
        x_shape = self.op.block.var(self.op.input_arg_names[2]).shape
        print(index_shape, out_grad_shape, x_shape) 
        tensor_zeros = core.GEOperatorFactory.create_operator("zeroslike" + self.getid(), "ZerosLike").set_input("x", x)    
        gather_out_grad = core.GEOperatorFactory.create_operator("gather" + self.getid(), "Gather").set_input("x", out_grad).set_input("indices", index).set_attr_bool("validate_indices", True)
        
        #x_var = core.GEOperatorFactory.create_operator("variable" + self.getid(), "Variable").set_attr_tensor("value", tensor_zeros)
        #index_var = core.GEOperatorFactory.create_operator("variable" + self.getid(), "Variable").set_attr_tensor("value", index)#.set_input("x", index)
        #updates_var = core.GEOperatorFactory.create_operator("variable" + self.getid(), "Variable").set_attr_tensor("value", gather_out_grad) #.set_input("x", gather_out_grad)
        
        #index = core.GEOperatorFactory.create_operator("cast" + self.getid(), "Cast").set_input("x", index).set_attr_int32("dst_type", 5)
        
        x_var = self.get_variable([3,x_shape[-1]], "float32", tensor_zeros)
        index_var = self.get_variable([3], "int32", index)
        updates_var = self.get_variable([3,x_shape[-1]], "float32", gather_out_grad)

        x_grad = core.GEOperatorFactory.create_operator("scatter" + self.getid(), "ScatterUpdate").set_input("var", x_var).set_input("indices", index_var).set_input("updates", updates_var)  
        #x_grad = out_grad

        return [tensor_zeros, gather_out_grad, x_var, index_var, updates_var, x_grad]

class TransposeGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TransposeGradParser, self).__init__(graph, var2geop)
        self.parser_name = "transpose2_grad"

    def _apply(self):
        print("===========", self.op.input_arg_names)
        out_grad = self.get_ge_input(self.op.input_arg_names[0])
        x_shape = self.get_ge_input(self.op.input_arg_names[1])
        perm_tensor = self.op.block.var(self.op.input_arg_names[1])
        perm_tensor_shape = perm_tensor.shape[1:] ## del the frst dim
        print(perm_tensor, perm_tensor_shape)
        
        #transpose = core.GEOperatorFactory.create_operator("transpose" + self.getid(), "TransposeD").set_input("x", data_x1_shape).set_attr_vec_int32("perm", perm)
        tensor = self.create_ge_tensor([len(perm_tensor_shape)], 2, perm_tensor_shape)
        const_shape = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Const").set_attr_tensor("value", tensor) 
        x_grad = core.GEOperatorFactory.create_operator("reshape" + self.getid(), "Reshape").set_input("x", out_grad).set_input("shape", const_shape)#.set_attr_int32("axis", axis)
        return [x_grad]

class LayerNormGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LayerNormGradParser, self).__init__(graph, var2geop)
        self.parser_name = "layer_norm_grad"

    def _apply(self):
        print("===========", self.op.input_arg_names)
        print("==========", self.op.output_arg_names)
        bias = self.get_ge_input(self.op.input_arg_names[0])
        mean = self.get_ge_input(self.op.input_arg_names[1])
        scale = self.get_ge_input(self.op.input_arg_names[2])
        variance = self.get_ge_input(self.op.input_arg_names[3])
        x = self.get_ge_input(self.op.input_arg_names[4])
        out_grad = self.get_ge_input(self.op.input_arg_names[5])

        x_grad = core.GEOperatorFactory.create_operator(self.parser_name + self.getid(), "LayerNormGrad").set_input("dy", out_grad).set_input("x", x).set_input("variance", variance).set_input("mean", mean).set_input("gamma", scale) 
        return [x_grad, x_grad, x_grad]






class TanhGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TanhGradParser, self).__init__(graph, var2geop)
        self.parser_name = 'tanh_grad'

    def _apply(self):
        print('TanhGradParser',self.op.input_arg_names)
        print('TanhGradParser',self.op.output_arg_names)
        y = self.get_ge_input(self.op.input_arg_names[0])
        dy = self.get_ge_input(self.op.input_arg_names[1])
        tanh_grad = core.GEOperatorFactory.create_operator("tanh_grad" + self.getid(), "TanhGrad").set_input("y", y).set_input("dy", dy)
        return [tanh_grad]

class LogGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(LogGradParser, self).__init__(graph, var2geop)
        self.parser_name = 'log_grad'

    def _apply(self):
        print('LogGradParser',self.op.input_arg_names)
        print('LogGradParser',self.op.output_arg_names)
        grad = self.get_ge_input(self.op.input_arg_names[0])
        input = self.get_ge_input(self.op.input_arg_names[1])
        log_grad = core.GEOperatorFactory.create_operator("log_grad" + self.getid(), "DivNoNan").set_input("x1", grad).set_input("x2", input)
        return [log_grad]

class SqrtGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SqrtGradParser, self).__init__(graph, var2geop)
        self.parser_name = "sqrt_grad"

    def _apply(self):
        # inputs={Out=['sqrt_0.tmp_0'], Out@GRAD=['sqrt_0.tmp_0@GRAD']
        y = self.get_ge_input(self.op.input_arg_names[0])
        dy = self.get_ge_input(self.op.input_arg_names[1])
        sqrt_grad = core.GEOperatorFactory.create_operator("sqrt_grad" + self.getid(), "SqrtGrad").set_input("y", y).set_input("dy", dy)
        return [sqrt_grad]

class PowGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(PowGradParser, self).__init__(graph, var2geop)
        self.parser_name = "pow_grad"

    def _apply(self):
        # Out@GRAD=['pow_0.tmp_0@GRAD'], X=['fc_0.tmp_1']}, factor = 2.0
        grad = self.get_ge_input(self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[1])
        factor = self.op.attr("factor")
        # print('-'*20)
        # print(x.shape)
        shape_tensor = self.create_shape_tensor()
        shape_tensor = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Shape").set_input("x", x)
        factor_scale = self.create_ge_tensor([1], 5, factor)
        factor_scale = core.GEOperatorFactory.create_operator("const" + self.getid(), "Const").set_attr_tensor("value", factor_scale)
        factor_tensor = core.GEOperatorFactory.create_operator("broadcast_to_d" + self.getid(), "BroadcastTo").set_input("x", factor_scale).set_input("shape", shape_tensor)

        x_power = core.GEOperatorFactory.create_operator("x_power" + self.getid(), "Power").set_input("x", x).set_attr_float("power", factor - 1)
        x_power_mul_factor = core.GEOperatorFactory.create_operator("x_power_mul_factor" + self.getid(), "Mul").set_input("x1", x).set_input("x2", factor_tensor)
        x_power_mul_factor_grad = core.GEOperatorFactory.create_operator("x_power_mul_factor_grad" + self.getid(), "Mul").set_input("x1", x_power_mul_factor).set_input("x2", grad)
        return [shape_tensor, factor_scale, factor_tensor, x_power, x_power_mul_factor, x_power_mul_factor_grad]

class GeluGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(GeluGradParser, self).__init__(graph, var2geop)
        self.parser_name = "gelu_grad"

    def _apply(self):
        # Out@GRAD=['gelu_0.tmp_0@GRAD'], X=['fc_0.tmp_1']
        print(self.op.input_arg_names)
        grad = self.get_ge_input(self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[1])

        y = core.GEOperatorFactory.create_operator("gelu" + self.getid(), "Gelu").set_input("x", x)
        gelu_grad = core.GEOperatorFactory.create_operator("gelu_grad" + self.getid(), "GeluGrad").set_input("x", x).set_input("dy", grad).set_input("y", y)
        return [gelu_grad]


class MeanGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MeanGradParser, self).__init__(graph, var2geop)
        self.parser_name = "mean_grad"

    def _apply(self):
        # mean_grad(inputs={Out@GRAD=['mean_0.tmp_0@GRAD'], X=['fc_0.tmp_1']}
        grad = self.get_ge_input(self.op.input_arg_names[0])
        x = self.get_ge_input(self.op.input_arg_names[1])

        # reshape grad to same dim with x
        shape_tensor = self.create_shape_tensor()
        shape_tensor = core.GEOperatorFactory.create_operator("shape" + self.getid(), "Shape").set_input("x", x)
        grad_reshape = core.GEOperatorFactory.create_operator("broadcast_to_d" + self.getid(), "BroadcastTo").set_input("x", grad).set_input("shape", shape_tensor)

        # # get grad of x
        ones_tensor = core.GEOperatorFactory.create_operator("one_tensor" + self.getid(), "OnesLike").set_input("x", x)
        sum = core.GEOperatorFactory.create_operator("mean" + self.getid(), "ReduceSumD").set_input("x", ones_tensor).set_attr_bool("keep_dims", False).set_attr_vec_int32("axes", [])
        mean = core.GEOperatorFactory.create_operator("x_power" + self.getid(), "Power").set_input("x", sum).set_attr_float("power", -1)
        mean_broadcast = core.GEOperatorFactory.create_operator("broadcast_to_d" + self.getid(), "BroadcastTo").set_input("x", mean).set_input("shape", shape_tensor)

        # # mul grad of x and grad
        mean_grad = core.GEOperatorFactory.create_operator("mean_grad" + self.getid(), "Mul").set_input("x1", mean_broadcast).set_input("x2", grad_reshape)

        return [shape_tensor, grad_reshape, ones_tensor, sum, mean, mena_broadcast, mean_grad]


class SGDParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SGDParser, self).__init__(graph, var2geop)
        self.parser_name = "sgd"

    def _apply(self):
        grad = self.get_ge_input(self.op.input_arg_names[0])
        lr = self.get_ge_input(self.op.input_arg_names[1])
        param = self.get_ge_input(self.op.input_arg_names[2])
        sgd = core.GEOperatorFactory.create_operator("momentum" + self.getid(), "ApplyGradientDescent").set_input("var", param).set_input("alpha", lr).set_input("delta", grad)
        return [sgd]


class AdamParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AdamParser, self).__init__(graph, var2geop)
        self.parser_name = "adam"

    def _apply(self):
        grad = self.get_ge_input(self.op.input_arg_names[0])
        lr = self.get_ge_input(self.op.input_arg_names[1])
        param = self.get_ge_input(self.op.input_arg_names[2])
        sgd = core.GEOperatorFactory.create_operator("momentum" + self.getid(), "ApplyGradientDescent").set_input("var", param).set_input("alpha", lr).set_input("delta", grad)
        return [sgd]







