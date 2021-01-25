"""
-*- coding:utf-8 -*-
"""
import os
import random
import numpy as np
import scipy.stats as stats
import ge


def check_ret(message, ret):
    """
    check return value
    """
    if ret != 0:
        raise Exception("{} failed ret = {}".format(message, ret))


def check_equal(message, val1, val2):
    """
    check return value
    """
    if val1 != val2:
        raise Exception("{} : {} != {}".format(message, val1, val2))


class PyGe(object):
    """
    class PyGe
    function: encapsulating methods of GE
    """
    def __init__(self, config, options):
        ret = ge.ge_initialize(config)
        check_ret("ge_initialize", ret)
        print("Initialize ge success.")

        self.session = ge.Session(options)
        if self.session:
            print("Create session success.")
        else:
            print("Create session fail.")
        # ge parameter initial
        self.ge_param = {
            "sample": {
                "batch_size": 1, "h": 28, "w": 28, "channel": 1
            },
            "conv2D_32": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "w": None, "lr": 0.2, "b": None,
                "strides": (1, 1, 1, 1), "pads": (2, 2, 2, 2), 
                "in_shape": [1, 28, 28, 1], "out_shape": [1, 28, 28, 32],
                "in_shape_tensor": None, 
                "w_shape": [32, 5, 5, 1], "b_shape": [25088, ], "w_shape_tensor": None,
                "var_name": ["conv2d_0_w", "conv2d_0_lr", "conv2d_0_b"],
                "var_value": None, "var_desc": None, "var_tensor": None,
                "x": None, "x_act": None,
                "var_w": None, "var_b": None
            },
            "conv2D_64": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "w": None, "lr": 0.2, "b": None, "strides": (1, 1, 1, 1), "pads": (2, 2, 2, 2),
                "in_shape": [1, 14, 14, 32], "out_shape": [1, 14, 14, 64],
                "in_shape_tensor": None,
                "w_shape": [64, 5, 5, 32], "b_shape": [12544, ], "w_shape_tensor": None,
                "var_name": ["conv2d_1_w", "conv2d_1_lr", "conv2d_1_b"],
                "var_value": None, "var_desc": None, "var_tensor": None,
                "x": None, "x_act": None,
                "var_w": None, "var_b": None
            },
            "pool": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "in_shape_32": [1, 28, 28, 32], "in_shape_64": [1, 14, 14, 64],
                "out_shape_32": [1, 14, 14, 32], "out_shape_64": [1, 7, 7, 64],
                "strides": (1, 2, 2, 1), "padding": "VALID",
                "ksize_32": (1, 2, 2, 1), "ksize_64": (1, 2, 2, 1),
                "shape_tensor_32": None, "shape_tensor_64": None,
                "x_32": None, "y_32": None, "x_64": None, "y_64": None
            },
            "fc_1024": {
                "in_shape": [1, 3136], "out_shape": [1, 3136],
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "w": None, "w_shape": [3136, 1024], "b": None, "b_shape": [1024, ], "lr": 0.2,
                "var_name": ["fc_0_w", "fc_0_b"],
                "var_value": None, "var_desc": None, "var_tensor": None, 
                "x": None, "x_act": None, "y_act": None,
                "var_w": None, "var_b": None
            },
            "fc_10": {
                "in_shape": [1, 1024], "out_shape": [1, 10],
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16,
                "w": None, "w_shape": [1024, 10], "b": None, "b_shape": [10, ], "lr": 0.2,
                "var_name": ["fc_1_w", "fc_1_b"],
                "var_value": None, "var_desc": None, "var_tensor": None,
                "x": None, "x_act": None, "y_act": None,
                "var_w": None, "var_b": None
            },
            "softmax": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16
            },
            "cross_entropy": {
                "fmt": ge.FORMAT_NHWC, "dt": ge.DT_FLOAT16
            }
        }

    def __del__(self):
        ret = ge.ge_finalize()
        check_ret("ge_finalize", ret)
        print("Finalize ge success.")
        del self.session
        print("free session success.")

    def init_sample_param(self, h, w, channel, batch_size=1):
        self.ge_param["sample"]["batch_size"] = batch_size
        self.ge_param["sample"]["h"] = h
        self.ge_param["sample"]["w"] = w
        self.ge_param["sample"]["channel"] = channel

    def update_sample_param(self, batch_size=1):
        self.ge_param["sample"]["batch_size"] = batch_size
        key = "conv2D_32"
        self.ge_param[key]["in_shape"][0] = self.ge_param["sample"]["batch_size"]
        in_shape = self.ge_param[key]["in_shape"]
        self.ge_param[key]["in_shape_tensor"] = self.gen_tensor([4], in_shape, fmt=self.ge_param[key]["fmt"],
                                                                dt=ge.DT_INT32)
        key = "conv2D_64"
        self.ge_param[key]["in_shape"][0] = self.ge_param["sample"]["batch_size"]
        in_shape = self.ge_param[key]["in_shape"]
        self.ge_param[key]["in_shape_tensor"] = self.gen_tensor([4], in_shape, fmt=self.ge_param[key]["fmt"],
                                                                dt=ge.DT_INT32)
        key = "fc_1024"
        self.ge_param[key]["in_shape"][0] = self.ge_param["sample"]["batch_size"]
        key = "fc_10"
        self.ge_param[key]["in_shape"][0] = self.ge_param["sample"]["batch_size"]

        self.init_pool_param()

    def save_paras(self):
        key = "conv2D_32"
        self.ge_param[key]["w"].tofile("./data/{}_w_1.bin".format(key))
        self.ge_param[key]["b"].tofile("./data/{}_b_1.bin".format(key))
        key = "conv2D_64"
        self.ge_param[key]["w"].tofile("./data/{}_w_1.bin".format(key))
        self.ge_param[key]["b"].tofile("./data/{}_b_1.bin".format(key))
        key = "fc_1024"
        self.ge_param[key]["w"].tofile("./data/{}_w_1.bin".format(key))
        self.ge_param[key]["b"].tofile("./data/{}_b_1.bin".format(key))
        key = "fc_10"
        self.ge_param[key]["w"].tofile("./data/{}_w_1.bin".format(key))
        self.ge_param[key]["b"].tofile("./data/{}_b_1.bin".format(key))
    
    def init_conv2d_param(self, filters, load_from_bin=False):
        key = "conv2D_" + str(filters)
        self.ge_param[key]["in_shape"][0] = self.ge_param["sample"]["batch_size"]
        w_shape = self.ge_param[key]["w_shape"]
        in_shape = self.ge_param[key]["in_shape"]
        b_shape = self.ge_param[key]["b_shape"]
        self.ge_param[key]["w_shape_tensor"] = self.gen_tensor([len(self.ge_param[key]['w_shape']), ],
                                               self.ge_param[key]['w_shape'], 
                                               fmt=self.ge_param[key]["fmt"], 
                                               dt=ge.DT_INT32)
        if load_from_bin:
            if os.path.exists('./data/' + key + '_w_1.bin'):
                self.ge_param[key]["b"] = np.fromfile('./data/' + key + '_b_1.bin', dtype=np.float16)
                self.ge_param[key]["w"] = np.fromfile('./data/' + key + '_w_1.bin', dtype=np.float16)
            else:
                self.ge_param[key]["b"] = np.fromfile('./data/' + key + '_b_0.bin', dtype=np.float16)
                self.ge_param[key]["w"] = np.fromfile('./data/' + key + '_w_0.bin', dtype=np.float16)
        else:
            self.ge_param[key]["w"] = self.gen_tensor_data_normal(w_shape, dt=self.ge_param[key]["dt"])
            self.ge_param[key]["b"] = self.gen_tensor_data(b_shape, value=0.02, dt=self.ge_param[key]["dt"])

        print(key, self.ge_param[key]["w"])

        self.ge_param[key]["strides"] = (1, 1, 1, 1)
        self.ge_param[key]["pads"] = (2, 2, 2, 2)
        self.ge_param[key]["in_shape_tensor"] = self.gen_tensor([4], in_shape, fmt=self.ge_param[key]["fmt"],
                                                         dt=ge.DT_INT32)
        tensor_desc = ge.TensorDesc(ge.Shape(w_shape), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])
        rate_desc = ge.TensorDesc(ge.Shape([1, ]), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])
        b_desc = ge.TensorDesc(ge.Shape(b_shape), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])
        self.ge_param[key]["var_desc"] = [tensor_desc, rate_desc, b_desc]
        self.ge_param[key]["var_value"] = [self.ge_param[key]["w"], self.ge_param[key]["lr"], self.ge_param[key]["b"]]
        self.ge_param[key]["var_tensor"] = [
            self.gen_tensor(w_shape, self.ge_param[key]["w"], fmt=self.ge_param[key]["fmt"],
                            dt=self.ge_param[key]["dt"]),
            self.gen_tensor([1, ], self.ge_param[key]["lr"], fmt=self.ge_param[key]["fmt"],
                            dt=self.ge_param[key]["dt"]),
            self.gen_tensor(b_shape, self.ge_param[key]["b"], fmt=self.ge_param[key]["fmt"],
                            dt=self.ge_param[key]["dt"]),
        ]

    def init_fc_param(self, n_out, load_from_bin=False):
        key = "fc_" + str(n_out)
        self.ge_param[key]["in_shape"][0] = self.ge_param["sample"]["batch_size"]
        w_shape = self.ge_param[key]["w_shape"]
        b_shape = self.ge_param[key]["b_shape"]

        if load_from_bin:
            if os.path.exists('./data/' + key + '_w_1.bin'):
                self.ge_param[key]["b"] = np.fromfile('./data/' + key + '_b_1.bin', dtype=np.float16)
                self.ge_param[key]["w"] = np.fromfile('./data/' + key + '_w_1.bin', dtype=np.float16)
            else:
                self.ge_param[key]["b"] = np.fromfile('./data/' + key + '_b_0.bin', dtype=np.float16)
                self.ge_param[key]["w"] = np.fromfile('./data/' + key + '_w_0.bin', dtype=np.float16)
        else:
            self.ge_param[key]["b"] = self.gen_tensor_data(b_shape, value=0.02, dt=self.ge_param[key]["dt"])
            self.ge_param[key]["w"] = self.gen_tensor_data_normal(w_shape, dt=self.ge_param[key]["dt"])

        print(key, self.ge_param[key]["w"])

        w_desc = ge.TensorDesc(ge.Shape(w_shape), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])
        b_desc = ge.TensorDesc(ge.Shape(b_shape), self.ge_param[key]["fmt"], self.ge_param[key]["dt"])
        self.ge_param[key]["var_desc"] = [w_desc, b_desc]
        self.ge_param[key]["var_value"] = [self.ge_param[key]["w"], self.ge_param[key]["b"]]
        self.ge_param[key]["var_tensor"] = [
                self.gen_tensor(w_shape, self.ge_param[key]["w"], fmt=self.ge_param[key]["fmt"],
                                dt=self.ge_param[key]["dt"]),
                self.gen_tensor(b_shape, self.ge_param[key]["b"], fmt=self.ge_param[key]["fmt"],
                                dt=self.ge_param[key]["dt"])
            ]
        
    def init_pool_param(self):
        key = "pool"
        self.ge_param[key]["in_shape_32"][0] = self.ge_param["sample"]["batch_size"]
        self.ge_param[key]["in_shape_64"][0] = self.ge_param["sample"]["batch_size"]
        self.ge_param[key]["out_shape_64"][0] = self.ge_param["sample"]["batch_size"] 
        self.ge_param[key]["out_shape_32"][0] = self.ge_param["sample"]["batch_size"]                             
        self.ge_param[key]["shape_tensor_32"] = [
            self.gen_tensor([len(self.ge_param[key]["in_shape_32"]),], 
                             self.ge_param[key]["in_shape_32"], fmt=self.ge_param[key]["fmt"],
                             dt=ge.DT_INT32),
            self.gen_tensor([len(self.ge_param[key]["out_shape_32"]),], 
                             self.ge_param[key]["out_shape_32"], fmt=self.ge_param[key]["fmt"],
                             dt=ge.DT_INT32)
        ]
        self.ge_param[key]["shape_tensor_64"] = [
            self.gen_tensor([len(self.ge_param[key]["in_shape_64"]),], 
                             self.ge_param[key]["in_shape_64"], fmt=self.ge_param[key]["fmt"],
                             dt=ge.DT_INT32),
            self.gen_tensor([len(self.ge_param[key]["out_shape_64"]),], 
                             self.ge_param[key]["out_shape_64"], fmt=self.ge_param[key]["fmt"],
                             dt=ge.DT_INT32)                             
        ]           

    def update_conv2d_params(self, filters, dw, db):
        key = "conv2D_" + str(filters)
        self.ge_param[key]['lr'] = 0.005
        self.ge_param[key]["w"] -= self.ge_param[key]['lr']*self.get_tensor_data(dw)
        self.ge_param[key]["b"] -= self.ge_param[key]['lr']*self.get_tensor_data(db)/self.ge_param[
            "sample"]["batch_size"]
        self.ge_param[key]["var_tensor"][0] = self.gen_tensor(self.ge_param[key]['w_shape'], 
                                                              self.ge_param[key]["w"], 
                                                              fmt=self.ge_param[key]["fmt"],
                                                              dt=self.ge_param[key]["dt"])
        self.ge_param[key]["var_tensor"][2] = self.gen_tensor(self.ge_param[key]['b_shape'], 
                                                              self.ge_param[key]["b"], 
                                                              fmt=self.ge_param[key]["fmt"],
                                                              dt=self.ge_param[key]["dt"])
        
    def update_fc_params(self, n_out, dw, db):
        key = "fc_" + str(n_out)
        self.ge_param[key]['lr'] = 0.005
        self.ge_param[key]["w"] -= self.ge_param[key]['lr']*self.get_tensor_data(dw)
        self.ge_param[key]["b"] -= self.ge_param[key]['lr']*self.get_tensor_data(db)/self.ge_param[
            "sample"]["batch_size"]
        self.ge_param[key]["var_tensor"][0] = self.gen_tensor(self.ge_param[key]['w_shape'], 
                                                              self.ge_param[key]["w"], 
                                                              fmt=self.ge_param[key]["fmt"],
                                                              dt=self.ge_param[key]["dt"])
        self.ge_param[key]["var_tensor"][1] = self.gen_tensor(self.ge_param[key]['b_shape'], 
                                                              self.ge_param[key]["b"], 
                                                              fmt=self.ge_param[key]["fmt"],
                                                              dt=self.ge_param[key]["dt"])

    def update_net_params(self, backward_outputs):
        # backward: fc_10_dw, fc_10_db, fc_1024_dw, fc_1024_db, conv2d_64_dw, conv2d_32_dw, conv2d_64_db, conv2d_32_db
        self.update_fc_params(10, backward_outputs[0], backward_outputs[1])
        self.update_fc_params(1024, backward_outputs[2], backward_outputs[3])
        self.update_conv2d_params(64, backward_outputs[4], backward_outputs[6])
        self.update_conv2d_params(32, backward_outputs[5], backward_outputs[7]) 
    
    def get_data_type_size(self, dt):
        """
        get data type size
        """
        dailation = 1
        if dt == ge.DT_FLOAT:
            dailation = 4
        elif dt == ge.DT_FLOAT16:
            dailation = 2
        elif dt == ge.DT_INT16:
            dailation = 2
        elif dt == ge.DT_UINT16:
            dailation = 2
        elif dt == ge.DT_INT32:
            dailation = 4
        elif dt == ge.DT_UINT32:
            dailation = 4
        elif dt == ge.DT_INT64:
            dailation = 8
        elif dt == ge.DT_UINT64:
            dailation = 8
        elif dt == ge.DT_INT8:
            dailation = 1
        return dailation

    def prints(self, msg, print_tensor, data_type, print_data=False):
        data = self.get_tensor_data(print_tensor, data_type)
        print('{}::'.format(msg), len(data), data.shape, print_tensor.get_size(), 
              print_tensor.get_tensor_desc().get_data_type(),
              print_tensor.get_tensor_desc().get_shape().get_dims())
        if print_data:
            print(data)

    def reshape(self, tensor, shape):
        tensor_data = np.array(tensor.get_data(), dtype=np.uint8)
        tensor_desc = ge.TensorDesc(ge.Shape(shape),
                                    tensor.get_tensor_desc().get_format(),
                                    tensor.get_tensor_desc().get_data_type())
        tensor.set_tensor_desc(tensor_desc)
    
    def get_tensor_data(self, tensor, dtype=np.float16):
        data = np.array(tensor.get_data(), dtype=np.uint8)
        b_arr = data.tobytes()
        arr_2 = np.frombuffer(b_arr, dtype=dtype)
        return arr_2
    
    def get_tensor_from_bin(self, in_path, shape_list, fmt=ge.FORMAT_ND, dt=ge.DT_FLOAT16):
        size = 1
        for i in range(len(shape_list)):
            size *= shape_list[i]
        data_len = size * self.get_data_type_size(dt)
        np_in = np.fromfile(in_path, dtype=np.uint8)
        np_size = np_in.size * np_in.itemsize

        check_equal("get_tensor_from_bin:", np_size, data_len)
        input_tensor_desc = ge.TensorDesc(ge.Shape(shape_list), fmt, dt)
        input_tensor_desc.set_real_dim_cnt(len(shape_list))
        input_tensor = ge.Tensor(input_tensor_desc, np_in)
        return input_tensor

    def remove_graph(self, graph_id):
        ret = self.session.remove_graph(graph_id)
        check_ret("remove_graph", ret)

    def add_graph(self, graph_id, graph):
        ret = self.session.add_graph(graph_id, graph)
        check_ret("add_graph", ret)

    def run_graph(self, graph_id, in_tensor, is_rebuild=False):
        out_tensor, ret = self.session.run_graph(graph_id, in_tensor)
        check_ret("run_graph", ret)
        if is_rebuild:
            ret = self.session.remove_graph(graph_id)
            check_ret("remove_graph", ret)
        return out_tensor

    def gen_tensor(self, tensor_shape, tensor_data, fmt=ge.FORMAT_NCHW, dt=ge.DT_FLOAT16):
        if dt == ge.DT_FLOAT:
            data_type = np.float32
        elif dt == ge.DT_FLOAT16:
            data_type = np.float16
        elif dt == ge.DT_INT32:
            data_type = np.int32
        else:
            data_type = np.int8
        
        np_data = np.array(tensor_data, dtype=data_type)
        np_data = np.frombuffer(np_data.tobytes(), dtype=np.uint8)
        input_tensor_desc = ge.TensorDesc(ge.Shape(tensor_shape), fmt, dt)
        tensor = ge.Tensor(input_tensor_desc, np_data)

        return tensor

    def gen_tensor_data(self, tensor_shape, value=None, dt=ge.DT_FLOAT16, l=-0.01, h=0.015):
        size = 1
        for i in range(len(tensor_shape)):
            size *= tensor_shape[i]

        if dt == ge.DT_FLOAT:
            data_type = np.float32
        elif dt == ge.DT_FLOAT16:
            data_type = np.float16
        else:
            data_type = np.int8
        np_data = np.zeros(size, dtype=data_type)
        if value:
            for i in range(size):
                np_data[i] = value
        else:
            for i in range(size):
                np_data[i] = random.uniform(l, h)
        return np_data

    def gen_tensor_data_normal(self, tensor_shape, dt=ge.DT_FLOAT16, mu=0, sigma=0.02):
        size = 1
        for i in range(len(tensor_shape)):
            size *= tensor_shape[i]
        
        if dt == ge.DT_FLOAT:
            data_type = np.float32
        elif dt == ge.DT_FLOAT16:
            data_type = np.float16
        else:
            data_type = np.int8

        np_data = np.zeros(size, dtype=data_type)
        lower, upper = mu - 2 * sigma, mu + 2 * sigma
        normal = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        return np.array(normal.rvs(size), dtype=data_type)

    def update_op_format(self, op, fmt):
        tensor_desc_x = op.get_input_desc("x")
        tensor_desc_y = op.get_output_desc("y")
        tensor_desc_x.set_format(fmt)
        tensor_desc_y.set_format(fmt)
        op.update_input_desc("x", tensor_desc_x)
        op.update_output_desc("y", tensor_desc_y)

    def update_op_datatype(self, op, dt):
        tensor_desc_x = op.get_input_desc("x")
        tensor_desc_y = op.get_output_desc("y")
        tensor_desc_x.set_data_type(dt)
        tensor_desc_y.set_data_type(dt)
        op.update_input_desc("x", tensor_desc_x)
        op.update_output_desc("y", tensor_desc_y)

    def update_op_desc(self, op, key):
        self.update_op_format(op, self.ge_param[key]["fmt"])
        self.update_op_datatype(op, self.ge_param[key]["dt"])

    def ge_reshape(self, x, shape, name):
        in_shape_tensor = self.gen_tensor([len(shape),], shape, fmt=ge.FORMAT_ND, dt=ge.DT_INT32)
        const_shape = ge.OperatorFactory.create_operator(name + '_const_shape',
                                                         "Constant").set_attr_tensor("value", in_shape_tensor)
        const_shape.update_output_desc("y", in_shape_tensor.get_tensor_desc())
        reshape = ge.OperatorFactory.create_operator(name + "_reshape", "Reshape").set_input("x", x) \
            .set_input("shape", const_shape)
        return reshape
    
    def ge_flatten(self, x, name):
        reshape = ge.OperatorFactory.create_operator(name + "_flatten", "Flatten").set_input("x", x)
        return reshape

    def ge_var_init(self, var_desc, var_name, var_tensor):
        var_desc.set_real_dim_cnt(var_desc.get_shape().get_dim_num())
        tmp_var_name = var_name + "_const"
        var_constant = ge.OperatorFactory.create_operator(tmp_var_name, "Constant").set_attr_tensor("value", var_tensor)
        var_constant.update_output_desc("y", var_desc)

        var_init = ge.OperatorFactory.create_operator(var_name, "Variable")
        var_init.update_output_desc("y", var_desc)
        tmp_var_name = var_name + "_assign"
        ge.OperatorFactory.create_operator(tmp_var_name, "Assign").set_input("ref", var_init).set_input(
            "value", var_constant)
        return var_init
    
    def init_graph(self, graph, var_descs, var_names, var_tensors):
        inputs, outputs = [], []
        for i in range(len(var_descs)):
            var_init = self.ge_var_init(var_descs[i], var_names[i], var_tensors[i])
            inputs.append(var_init)
        graph.set_inputs(inputs).set_outputs(outputs)

    def create_var(self, graph, var_desc, var_name):
        var_op_list = []
        for i in range(len(var_desc)):
            var_op = ge.OperatorFactory.create_operator(var_name[i], "Variable")
            var_op.update_output_desc("y", var_desc[i])
            graph.add_op(var_op)
            var_op_list.append(var_op)
        return var_op_list[0] if i == 0 else var_op_list

    def layer_conv2D(self, graph, x, filters):
        key = "conv2D_" + str(filters)
        var_desc = [self.ge_param[key]["var_desc"][0], self.ge_param[key]["var_desc"][2]]
        var_name = [self.ge_param[key]["var_name"][0], self.ge_param[key]["var_name"][2]]
        var_w, var_b = self.create_var(graph, var_desc, var_name)
        self.ge_param[key]["var_w"] = var_w
        self.ge_param[key]["var_b"] = var_b
        conv2d = ge.OperatorFactory.create_operator(key, "Conv2D") \
            .set_input("x", x) \
            .set_input("filter", var_w) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_vec_int64("pads", self.ge_param[key]["pads"]) \
            .set_input("bias", var_b)
        self.update_op_desc(conv2d, key)
        tensor_desc_w = conv2d.get_input_desc("filter")
        tensor_desc_w.set_format(self.ge_param[key]["fmt"])
        conv2d.update_input_desc("filter", tensor_desc_w)
        graph.add_op(conv2d)
        return conv2d

    def layer_pool(self, graph, name, x, ksize=(1, 2, 2, 1)):
        key = "pool"
        op_name = key + name
        avg_pool = ge.OperatorFactory.create_operator(op_name, "AvgPool") \
            .set_input("x", x) \
            .set_attr_vec_int64("ksize", ksize) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_string("padding", self.ge_param[key]["padding"])
        self.update_op_desc(avg_pool, key)
        graph.add_op(avg_pool)
        return avg_pool

    def layer_fc(self, graph, x, num_out=10, use_dropout=False, transpose_x1=False, transpose_x2=False):
        key = "fc_" + str(num_out)
        var_desc = self.ge_param[key]["var_desc"]
        var_name = self.ge_param[key]["var_name"]
        var_w, var_b = self.create_var(graph, var_desc, var_name)
        self.ge_param[key]["var_w"] = var_w
        self.ge_param[key]["var_b"] = var_b
        fc = ge.OperatorFactory.create_operator(key + "matmul_fc",
                                                "MatMul").set_input("x1", x) \
            .set_input("x2", var_w) \
            .set_input("bias", var_b) \
            .set_attr_bool("transpose_x1", transpose_x1) \
            .set_attr_bool("transpose_x2", transpose_x2)
        self.update_op_desc(fc, key)
        graph.add_op(fc)
        return fc

    def layer_softmax(self, graph, x):
        key = 'softmax'
        softmax = ge.OperatorFactory.create_operator("softmax",
                                                     "SoftmaxV2").set_input("x", x)
        self.update_op_desc(softmax, key)
        graph.add_op(softmax)
        return softmax

    def layer_loss(self, graph, features, labels):
        key = "cross_entropy"
        cross = ge.OperatorFactory.create_operator("softmax_cross_entropy_with_logits",
                                                   "SoftmaxCrossEntropyWithLogits").set_input("features", features) \
            .set_input("labels", labels)
        self.update_op_desc(cross, key)
        graph.add_op(cross)
        return cross

    def ge_relu(self, graph, name, x):
        # max(x, 0)
        relu = ge.OperatorFactory.create_operator(name + "_relu", "Relu").set_input("x", x)
        graph.add_op(relu)
        return relu

    def ge_relu_grad(self, graph, name, gradients, features):
        relu_grad = ge.OperatorFactory.create_operator(name + "_relu_grad", "ReluGrad") \
            .set_input("gradients", gradients) \
            .set_input("features", features)
        graph.add_op(relu_grad)
        return relu_grad

    def mnist_forward(self, graph, is_train=False):
        # conv2D 0 [N, 28, 28, 1] conv2d [32, 5, 5, 1] -> [N, 28, 28, 32]
        data_x = ge.OperatorFactory.create_operator("conv2d_0_x", "Data").set_attr_int64("index", 0)
        conv2d_0 = self.layer_conv2D(graph, data_x, 32)
        conv2d_0_relu = self.ge_relu(graph, "conv2d_0", conv2d_0)
        # pool 0 [N, 28, 28, 32] -> [N, 14, 14, 32]
        pool_0 = self.layer_pool(graph, "pool_0", conv2d_0_relu, ksize=self.ge_param["pool"]["ksize_32"])
        # conv2D 1 [N, 14, 14, 32] conv2d [64, 5, 5, 32] -> [N, 14, 14, 64]
        conv2d_1 = self.layer_conv2D(graph, pool_0, 64)
        conv2d_1_relu = self.ge_relu(graph, "conv2d_1", conv2d_1)
        # pool 1 [N, 14, 14, 64] -> [N, 7, 7, 64]
        pool_1 = self.layer_pool(graph, "pool_1", conv2d_1_relu, ksize=self.ge_param["pool"]["ksize_64"])
        # fc 256 [N, 7*7*64] [7*7*64, 256] -> [N, 256]
        reshape_pool = self.ge_flatten(pool_1, "fc_1024_forward/flatten")
        fc_1024 = self.layer_fc(graph, reshape_pool, num_out=1024, transpose_x1=0, transpose_x2=0)
        fc_1024_relu = self.ge_relu(graph, "fc_1024", fc_1024)
        # fc 10 [N, 256] [256, 10] -> [N, 10]
        fc_10 = self.layer_fc(graph, fc_1024_relu, num_out=10, transpose_x1=0, transpose_x2=0)
        # softmax [N, 10] -> [N, 10]
        if is_train:
            soft_m = self.layer_softmax(graph, fc_10)
            # cross entropy [N, 10] , [N, 10]
            data_labels = ge.OperatorFactory.create_operator("labels", "Data").set_attr_int64("index", 1)
            cross = self.layer_loss(graph, fc_10, data_labels)
            return {'data_x': data_x, 'loss': cross, 'conv2d_0': conv2d_0, 'pool_0': pool_0, 'conv2d_1': conv2d_1,
                    'pool_1':reshape_pool, 'fc_0': fc_1024, 'fc_0_relu': fc_1024_relu, 'softmax': soft_m}
        else:
            soft_m = self.layer_softmax(graph, fc_10)
            graph.set_inputs([data_x]).set_outputs([soft_m])

    def ge_matmul(self, name, x1, x2, idx_1=0, idx_2=0, b=None, transpose_x1=False, transpose_x2=False):
        matmul = ge.OperatorFactory.create_operator(name + "_matmul",
                                                    "MatMul").set_input("x1", x1, idx_1) \
            .set_input("x2", x2, idx_2) \
            .set_attr_bool("transpose_x1", transpose_x1) \
            .set_attr_bool("transpose_x2", transpose_x2)
        if b:
            matmul.set_input("bias", b)
        return matmul

    def layer_softmax_grad(self, graph, softmax, grad_softmax):
        key = "softmax"
        softmax_grad = ge.OperatorFactory.create_operator("soft_max_grad", "SoftmaxGrad") \
            .set_input("softmax", softmax) \
            .set_input("grad_softmax", grad_softmax)
        self.update_op_desc(softmax_grad, key)
        graph.add_op(softmax_grad)
        return softmax_grad

    def layer_fc_grad(self, graph, name, x1, x2, idx_1=0, idx_2=0, b=None, num_out=10, transpose_x1=False,
        transpose_x2=False):
        key = "fc_" + str(num_out)
        fc_grad = self.ge_matmul(key + '_' + name, x1, x2, idx_1, idx_2, b, transpose_x1=transpose_x1, 
            transpose_x2=transpose_x2)
        self.update_op_desc(fc_grad, key)
        graph.add_op(fc_grad)
        return fc_grad

    def layer_avgpool_grad(self, graph, name, filters, input_grad, ksize):
        key = "pool"
        in_shape = 'shape_tensor_' + str(filters)
        orig_input_shape = ge.OperatorFactory.create_operator(key + in_shape, "Constant") \
            .set_attr_tensor("value", self.ge_param[key][in_shape][0])
        orig_input_shape.update_output_desc("y", self.ge_param[key][in_shape][0].get_tensor_desc())
        graph.add_op(orig_input_shape)
        avg_pool_grad = ge.OperatorFactory.create_operator(name + "_avg_pool_grad", "AvgPoolGrad") \
            .set_input("orig_input_shape", orig_input_shape) \
            .set_input("input_grad", input_grad) \
            .set_attr_vec_int64("ksize", ksize) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_string("padding", self.ge_param[key]["padding"])
        self.update_op_desc(avg_pool_grad, key)
        graph.add_op(avg_pool_grad)
        return avg_pool_grad

    def layer_conv2D_filter_grad(self, graph, filters, x, out_backprop):
        key = "conv2D_" + str(filters)

        filter_size = ge.OperatorFactory.create_operator(key + "_grad_filter/filter_size", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["w_shape_tensor"])
        graph.add_op(filter_size)

        conv2d_grad_filter = ge.OperatorFactory.create_operator(key + "_filter_grad",
                                                                "Conv2DBackpropFilter") \
            .set_input("x", x) \
            .set_input("out_backprop", out_backprop) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_vec_int64("pads", self.ge_param[key]["pads"]) \
            .set_input("filter_size", filter_size)
        self.update_op_desc(conv2d_grad_filter, key)
        graph.add_op(conv2d_grad_filter)
        return conv2d_grad_filter

    def layer_conv2D_x_grad(self, graph, filters, input_size, filter_w, out_backprop):
        key = "conv2D_" + str(filters)
        conv2d_x_grad = ge.OperatorFactory.create_operator(key + "_x_grad",
                                                           "Conv2DBackpropInput") \
            .set_input("input_size", input_size) \
            .set_input("filter", filter_w) \
            .set_input("out_backprop", out_backprop) \
            .set_attr_vec_int64("strides", self.ge_param[key]["strides"]) \
            .set_attr_vec_int64("pads", self.ge_param[key]["pads"])
        self.update_op_desc(conv2d_x_grad, key)
        graph.add_op(conv2d_x_grad)
        return conv2d_x_grad

    def mnist_backprop(self, graph, para_forward):
        key = "softmax"
        softmax_grad = para_forward['loss']
        key = "fc_10"
        # fc 10
        fc_10_x1 = para_forward['fc_0_relu']
        # [N, 256].T x [N, 10] -> [256, 10]
        fc_10_dw = self.ge_matmul("fc_10_dw", fc_10_x1, softmax_grad, idx_2=1, transpose_x1=1, transpose_x2=0)
        # [1, N]  [N, 10] -> [1, 10]
        b_shape = [1, self.ge_param["sample"]["batch_size"]]
        b_data = self.gen_tensor_data(b_shape, value=1, dt=self.ge_param[key]["dt"])
        b_tensor = self.gen_tensor(b_shape, b_data, fmt=self.ge_param[key]["fmt"], dt=self.ge_param[key]["dt"])
        db_x1 = ge.OperatorFactory.create_operator("fc_db/x1", "Constant").set_attr_tensor("value", b_tensor)
        graph.add_op(db_x1)
        fc_10_db = self.ge_matmul("fc_10_db", db_x1, softmax_grad, idx_2=1, transpose_x1=0, transpose_x2=0)
        fc_10_w = self.ge_param[key]["var_w"]
        # [N, 10] x [256, 10] -> [N, 256]
        fc_10_grad = self.layer_fc_grad(graph, "fc_10_grad", softmax_grad, fc_10_w, idx_1=1, num_out=10,
            transpose_x1=0, transpose_x2=1)

        # fc 1024
        key = "fc_1024"
        fc_1024_x1 = para_forward['pool_1']
        # for relu
        fc_1024_x_act = para_forward['fc_0']
        #graph.add_op(fc_1024_x_act)
        # [N, 1024] × [N, 1024] ->
        fc_1024_relu_grad = self.ge_relu_grad(graph, "fc_1024", fc_10_grad, fc_1024_x_act)
        # [N, 3136] × [N, 1024] -> [3136, 1024]
        fc_1024_dw = self.ge_matmul("fc_1024_dw", fc_1024_x1, fc_1024_relu_grad, transpose_x1=1, transpose_x2=0)
        # [1, N] -> [1, 1024]
        fc_1024_db = self.ge_matmul("fc_1024_db", db_x1, fc_1024_relu_grad, transpose_x1=0, transpose_x2=0)
        fc_1024_w = self.ge_param[key]["var_w"]
        # [N, 1024] × [3136, 1024] -> [N, 3136]
        fc_1024_grad = self.layer_fc_grad(graph, "fc_1024_grad", fc_1024_relu_grad, fc_1024_w, num_out=1024,
            transpose_x1=0, transpose_x2=1)

        # pool 1
        key = "pool"
        # [N, 3136] -> [N, 7, 7, 64] -> [N, 14, 14, 64]
        input_shapes = self.ge_param[key]["out_shape_64"]
        fc_1024_grad_reshape = self.ge_reshape(fc_1024_grad, input_shapes, "pool_1_grad")
        graph.add_op(fc_1024_grad_reshape)
        pool_1_grad = self.layer_avgpool_grad(graph, "pool_1_grad", 64, fc_1024_grad_reshape, self.ge_param[
            "pool"]["ksize_64"])

        # conv2D 1
        key = "conv2D_64"
        conv2d_64_x = para_forward['pool_0']
        # relu
        conv2D_64_x_act = para_forward['conv2d_1']
        conv2D_64_relu_grad = self.ge_relu_grad(graph, "conv2D_64", pool_1_grad, conv2D_64_x_act)
        # [N, 14, 14, 64] x [64, 5, 5, 32] -> [N, 14, 14, 32]
        conv2d_64_dw = self.layer_conv2D_filter_grad(graph, 64, conv2d_64_x, conv2D_64_relu_grad)
        conv2D_64_relu_grad_flat = self.ge_flatten(conv2D_64_relu_grad, "conv2d_64_db/flatten")
        conv2d_64_db = self.ge_matmul("conv2d_64_db", db_x1, conv2D_64_relu_grad_flat, transpose_x1=0, transpose_x2=0)
        graph.add_op(conv2d_64_db)
        # [N, 28, 28, 64] x [64, 5, 5, 32]
        input_size_1 = ge.OperatorFactory.create_operator("conv2d_64_grad_in/input_size", "Constant") \
            .set_attr_tensor("value", self.ge_param[key]["in_shape_tensor"])
        input_size_1.update_output_desc("y", self.ge_param[key]["in_shape_tensor"].get_tensor_desc())
        graph.add_op(input_size_1)
        conv_w_1 = self.ge_param[key]["var_w"]
        conv2d_64_grad_in = self.layer_conv2D_x_grad(graph, 64, input_size_1, conv_w_1, conv2D_64_relu_grad)

        # pool 0
        key = "pool"
        # [N, 14, 14, 32] -> [N, 28, 28, 32]
        pool_0_grad = self.layer_avgpool_grad(graph, "pool_0_grad", 32, conv2d_64_grad_in,
            self.ge_param["pool"]["ksize_32"])
        # conv2D 0 : in = [N, 28, 28, 1], out = [32, 5, 5, 1]
        key = "conv2D_32"
        # relu
        conv2D_32_x_act = para_forward['conv2d_0']
        conv2D_32_relu_grad = self.ge_relu_grad(graph, "conv2D_32", pool_0_grad, conv2D_32_x_act)
        conv2d_32_x = para_forward['data_x']
        conv2d_32_dw = self.layer_conv2D_filter_grad(graph, 32, conv2d_32_x, conv2D_32_relu_grad)
        conv2D_32_relu_grad_flat = self.ge_flatten(conv2D_32_relu_grad, "conv2d_32_db/flatten")
        conv2d_32_db = self.ge_matmul("conv2d_32_db", db_x1, conv2D_32_relu_grad_flat, transpose_x1=0, transpose_x2=0)
        graph.add_op(conv2d_32_db)
        graph.set_inputs([para_forward['data_x']])\
             .set_outputs([fc_10_dw, fc_10_db, fc_1024_dw, fc_1024_db, conv2d_64_dw, conv2d_32_dw, conv2d_64_db,
                           conv2d_32_db, para_forward['softmax']])

    def construct_var_list(self):
        var_desc, var_name, var_tensor = [], [], []
        key = "conv2D_32"
        var_desc += [self.ge_param[key]["var_desc"][0], self.ge_param[key]["var_desc"][2]]
        var_name += [self.ge_param[key]["var_name"][0], self.ge_param[key]["var_name"][2]]
        var_tensor += [self.ge_param[key]["var_tensor"][0], self.ge_param[key]["var_tensor"][2]]
        key = "conv2D_64"
        var_desc += [self.ge_param[key]["var_desc"][0], self.ge_param[key]["var_desc"][2]]
        var_name += [self.ge_param[key]["var_name"][0], self.ge_param[key]["var_name"][2]]
        var_tensor += [self.ge_param[key]["var_tensor"][0], self.ge_param[key]["var_tensor"][2]]
        key = "fc_1024"
        var_desc += self.ge_param[key]["var_desc"]
        var_name += self.ge_param[key]["var_name"]
        var_tensor += self.ge_param[key]["var_tensor"]
        key = "fc_10"
        var_desc += self.ge_param[key]["var_desc"]
        var_name += self.ge_param[key]["var_name"]
        var_tensor += self.ge_param[key]["var_tensor"]
        return var_desc, var_name, var_tensor
