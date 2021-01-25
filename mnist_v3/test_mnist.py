"""
-*- coding:utf-8 -*-
"""
import os
import h5py
import matplotlib.image as mpimg
import numpy as np
import unittest
import ge
from mnist import PyGe

PATH = "./data/"
GRAPH_MNIST = 1
GRAPH_INIT = 0

def check_ret(message, ret):
    """
    check return value
    """
    if ret != 0:
        raise Exception("{} failed ret = {}".format(message, ret))

def get_class_methods(class_name):
    method_list = []
    for method in dir(class_name):
        str_list = method.split('_')
        if str_list[0] == 'test':
            method_list.append(str_list)

    method_list = sorted(method_list, key=lambda x: x[2])

    methods = []
    for method in method_list:
        methods.append("_".join(method))

    return methods

def get_test_case_by_list(suite, case_class, methods, listx):
    for case_no in listx:
        for method in methods:
            if method.find(case_no) >= 0:
                suite.addTest(case_class(method))
    return suite

def get_test_case_by_opt(suite, case_class, methods, opt):
    for method in methods:
        if method.find(opt) >= 0:
            suite.addTest(case_class(method))

    return suite

def switch_cases(case_class, opt):
    """
    :param case_class:
    :param opt:
        "all"   : all test cases
        "001"   : test case number 001
        "err"   : all error test cases
    :return:
    """
    suite = unittest.TestSuite()
    methods = get_class_methods(case_class)

    if type(opt) == list:
        return get_test_case_by_list(suite, case_class, methods, opt)

    if opt == "all":
        for method in methods:
            suite.addTest(case_class(method))
        return suite
    if opt == "err":
        suite = get_test_case_by_opt(suite, case_class, methods, opt)
    else:
        for method in methods:
            if method.find(opt) >= 0:
                suite.addTest(case_class(method))
                break
    return suite


class DataSet(object):
    def __init__(self, img_path, dataset_path, h5_file, x_key="train_set_x", y_key="train_set_y"):
        self.file = h5_file
        self.img_path = img_path
        self.dataset_path = dataset_path
        self.x_key = x_key
        self.y_key = y_key
        self.classes = "list_classes"

    def make_dataset_label_one_hot(self):
        x, y, classs = [], [], []
        for _, image_path in enumerate(os.listdir(self.img_path)):
            # label to one-hot
            label = int(image_path.split('_')[0])
            # label value
            label_one_hot = [0 if i != label else 1 for i in range(10)]
            y.append(label_one_hot)
            # label class
            label_one_hot = [0 if i != label else 1 for i in range(10)]
            classs.append(label_one_hot)

            # value
            path = self.img_path + '/{}'
            image = mpimg.imread(path.format(image_path))
            w = image.shape[0]
            h = image.shape[1]
            c = 1 if len(image.shape) == 2 else image.shape[2]
            # reshape
            np_image = 1 - np.reshape(image, [w, h, c])
            x.append(np_image)
            # print("data_set:::: \nlabels", label_one_hot, "\ndata", np.reshape(image, [w*h*c, ]))

        h5_file = self.dataset_path + '/' + self.file
        if os.path.exists(h5_file):
            os.remove(h5_file)
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset(self.x_key, data=np.array(x))
            f.create_dataset(self.y_key, data=np.array(y))
            f.create_dataset(self.classes, data=np.array(y))

    def load_dataset(self, train_file):
        train_dataset = h5py.File(train_file, "r")

        # features
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        # labels
        train_set_y = np.array(train_dataset["train_set_y"][:])
        # classes
        # train_classes = np.array(train_dataset["list_classes"][:])

        return train_set_x_orig, train_set_y

def build_dataset(ds_file):
    ds_mnist = DataSet("./train_images", "./data_set", ds_file)
    ds_mnist.make_dataset_label_one_hot()
    
    return ds_mnist.load_dataset("./data_set/" + ds_file)

def load_one_set(img_file):
    x = []
    image = mpimg.imread(img_file)
    w = image.shape[0]
    h = image.shape[1]
    c = 1 if len(image.shape) == 2 else image.shape[2]
    np_image = 1 - np.reshape(image, [w, h, c])
    x.append(np_image)
    return np.array(x)


class TestGe(unittest.TestCase):
    def debug_info(self, ge_handle, info_list, info):
        f = open("./debug.log", "a+")
        for i in range(len(info_list)):
            data = ge_handle.get_tensor_data(info[i])
            f.write('{}:: {} {} {}\n\n'.format(info_list[i], info[i].get_tensor_desc().get_shape().get_dims(), data.shape, data))
        f.close()

    def model(self, ge_handle, inputs):
        # forward and backward
        graph = ge.Graph("mnist")
        out_forward = ge_handle.mnist_forward(graph, is_train=True)
        ge_handle.mnist_backprop(graph, out_forward)
        ge_handle.add_graph(GRAPH_MNIST, graph)
        # init
        graph_init = ge.Graph("init_mnist")
        var_desc_init, var_name_init, var_tensor_init = ge_handle.construct_var_list()
        ge_handle.init_graph(graph_init, var_desc_init, var_name_init, var_tensor_init)
        ge_handle.add_graph(GRAPH_INIT, graph_init)
        # run
        ge_handle.run_graph(GRAPH_INIT, [], is_rebuild=True)
        outputs_forward = ge_handle.run_graph(GRAPH_MNIST, inputs)
        ge_handle.update_net_params(outputs_forward)

    def predict(self, ge_handle, inputs):
        # forward
        graph = ge.Graph("mnist")
        ge_handle.mnist_forward(graph)
        ge_handle.add_graph(GRAPH_MNIST, graph)
        # init
        graph_init = ge.Graph("init_mnist")
        var_desc_init, var_name_init, var_tensor_init = ge_handle.construct_var_list()
        ge_handle.init_graph(graph_init, var_desc_init, var_name_init, var_tensor_init)
        ge_handle.add_graph(GRAPH_INIT, graph_init)
        # run
        ge_handle.run_graph(GRAPH_INIT, [], is_rebuild=True)
        outputs_forward = ge_handle.run_graph(GRAPH_MNIST, inputs, is_rebuild=True)
        ge_handle.prints("predict", outputs_forward[0], np.float16, print_data=1)
        return ge_handle.get_tensor_data(outputs_forward[0])

    def train(self, ge_handle, x_split, y_split, batch=1, epoch=5, num_iterations=20):
        if os.path.exists("./debug.log"):
            os.remove("./debug.log")

        in_tensor_x = ge_handle.gen_tensor(x_split[0].shape, x_split[0], fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        in_tensor_y = ge_handle.gen_tensor(y_split[0].shape, y_split[0], fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        inputs = [in_tensor_x, in_tensor_y]

        self.model(ge_handle, inputs)

        for j in range(epoch):
            print("============ epoch {} ==================\n".format(j+1))
            for i in range(batch):
                in_tensor_x = ge_handle.gen_tensor(x_split[i].shape, x_split[i], fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
                in_tensor_y = ge_handle.gen_tensor(y_split[i].shape, y_split[i], fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
                real_batch_size = x_split[i].shape[0]
                inputs = [in_tensor_x, in_tensor_y]
                ge_handle.update_sample_param(batch_size=real_batch_size)
                for i in range(num_iterations):
                    print("============ train {} times ==================\n".format(i+1))
                    # init forward
                    graph_init = ge.Graph("init_mnist")
                    var_desc_init, var_name_init, var_tensor_init = ge_handle.construct_var_list()
                    ge_handle.init_graph(graph_init, var_desc_init, var_name_init, var_tensor_init)
                    ge_handle.add_graph(GRAPH_INIT, graph_init)
                    ge_handle.run_graph(GRAPH_INIT, [], is_rebuild=True)
                    # run mnist
                    outputs = ge_handle.run_graph(GRAPH_MNIST, inputs)
                    prints = []
                    data = ge_handle.get_tensor_data(outputs[8])
                    for j in range(data.size):
                        if j % 10 == 0 and j != 0:
                            prints += ['|']
                        prints += [data[j]]
                    print("softmax::", prints)
                    ge_handle.update_net_params(outputs)

    def test_000_dataset(self):
        train_set_x, train_set_y = build_dataset("train_mnist.h5")
        print(train_set_x.shape, train_set_y.shape)

    def test_001_mnist(self):
        """
        network mnist by GE
        |============= conv ============== | ===== pool ===== | ===== conv ===== | ===== pool ===== | ===== fc ===== | === softmax === |
        |input: 28×28×1 -> cells: 28×28×32 |  cells: 14×14×32 |  cells: 14×14×64 |  cells:  7×7×64  |   1×1×1024     |     1×1×10      |
        |  Conv     : 5×5                  |   Pool   : 2×2   |    conv   : 5×5  |   Pool   : 2×2   |                |                 |
        |  filters  : 32                   |   stride : 2     |    filter :64    |   stride : 2     |   dropout 0.5  |                 |
        |  padding  : 2                    |                  |    padding:2     |                  |                |                 |
        ================================================================================================================================
        """
        config = {"ge.exec.deviceId": "0", "ge.graphRunMode": "1", "ge.exec.precision_mode": "allow_mix_precision"}
        options = {}
        ge_handle = PyGe(config, options)

        train_set_x, train_set_y = build_dataset("train_mnist.h5")
        in_shape_x = train_set_x.shape
        in_shape_y = train_set_y.shape
        print(in_shape_x[0], in_shape_x, in_shape_y)

        # split sample : x (200, 28, 28, 1)  y (200, 10)
        batch_size = 100
        is_zero = in_shape_x[0] % batch_size
        n = in_shape_x[0] // batch_size
        num = n + 1 if is_zero else n

        np.random.seed(0)
        np.random.shuffle(train_set_x)
        np.random.seed(0)
        np.random.shuffle(train_set_y)

        x_split = np.array_split(train_set_x, num, axis=0)
        y_split = np.array_split(train_set_y, num, axis=0)

        real_batch_size = x_split[0].shape[0]
        ge_handle.init_sample_param(28, 28, 1, batch_size=real_batch_size)
        ge_handle.init_conv2d_param(32)
        ge_handle.init_conv2d_param(64)
        ge_handle.init_fc_param(1024)
        ge_handle.init_fc_param(10)
        ge_handle.init_pool_param()

        self.train(ge_handle, x_split, y_split, batch=num, epoch=2, num_iterations=50)

        # save w and b
        ge_handle.save_paras()
        ge_handle.remove_graph(GRAPH_MNIST)

        img_file = "./test_images/2_0.png"
        img_x = load_one_set(img_file)
        in_tensor = ge_handle.gen_tensor(img_x.shape, img_x, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        predict_inputs = [in_tensor]
        prediction = self.predict(ge_handle, predict_inputs)
        list_pre = prediction.tolist()
        print("{} prediction is {}".format(img_file, list_pre.index(max(list_pre))))

        img_file = "./test_images/0_11.png"
        img_x = load_one_set(img_file)
        in_tensor = ge_handle.gen_tensor(img_x.shape, img_x, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        predict_inputs = [in_tensor]
        prediction = self.predict(ge_handle, predict_inputs)
        list_pre = prediction.tolist()
        print("{} prediction is {}".format(img_file, list_pre.index(max(list_pre))))

        img_file = "./test_images/7_10.png"
        img_x = load_one_set(img_file)
        in_tensor = ge_handle.gen_tensor(img_x.shape, img_x, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        predict_inputs = [in_tensor]
        prediction = self.predict(ge_handle, predict_inputs)
        list_pre = prediction.tolist()
        print("{} prediction is {}".format(img_file, list_pre.index(max(list_pre))))

        img_file = "./test_images/9_35.png"
        img_x = load_one_set(img_file)
        in_tensor = ge_handle.gen_tensor(img_x.shape, img_x, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        predict_inputs = [in_tensor]
        prediction = self.predict(ge_handle, predict_inputs)
        list_pre = prediction.tolist()
        print("{} prediction is {}".format(img_file, list_pre.index(max(list_pre))))

    def test_002_mnist(self):
        config = {"ge.exec.deviceId": "0", "ge.graphRunMode": "1", "ge.exec.precision_mode": "allow_mix_precision"}
        options = {}
        ge_handle = PyGe(config, options)

        ge_handle.init_sample_param(28, 28, 1, batch_size=1)
        ge_handle.init_conv2d_param(32, load_from_bin=True)
        ge_handle.init_conv2d_param(64, load_from_bin=True)
        ge_handle.init_fc_param(1024, load_from_bin=True)
        ge_handle.init_fc_param(10, load_from_bin=True)
        ge_handle.init_pool_param()

        img_file = "./test_images/2_0.png"
        img_x = load_one_set(img_file)
        in_tensor = ge_handle.gen_tensor(img_x.shape, img_x, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        predict_inputs = [in_tensor]
        prediction = self.predict(ge_handle, predict_inputs)
        list_pre = prediction.tolist()
        print("{} prediction is {}".format(img_file, list_pre.index(max(list_pre))))

        img_file = "./test_images/0_11.png"
        img_x = load_one_set(img_file)
        in_tensor = ge_handle.gen_tensor(img_x.shape, img_x, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        predict_inputs = [in_tensor]
        prediction = self.predict(ge_handle, predict_inputs)
        list_pre = prediction.tolist()
        print("{} prediction is {}".format(img_file, list_pre.index(max(list_pre))))

        img_file = "./test_images/7_10.png"
        img_x = load_one_set(img_file)
        in_tensor = ge_handle.gen_tensor(img_x.shape, img_x, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        predict_inputs = [in_tensor]
        prediction = self.predict(ge_handle, predict_inputs)
        list_pre = prediction.tolist()
        print("{} prediction is {}".format(img_file, list_pre.index(max(list_pre))))

        img_file = "./test_images/9_35.png"
        img_x = load_one_set(img_file)
        in_tensor = ge_handle.gen_tensor(img_x.shape, img_x, fmt=ge.FORMAT_NHWC, dt=ge.DT_FLOAT16)
        predict_inputs = [in_tensor]
        prediction = self.predict(ge_handle, predict_inputs)
        list_pre = prediction.tolist()
        print("{} prediction is {}".format(img_file, list_pre.index(max(list_pre))))


if __name__ == "__main__":
    suite = switch_cases(TestGe, ["001", "002"])
    unittest.TextTestRunner(verbosity=2).run(suite)
