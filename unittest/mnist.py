# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import paddle
import paddle.fluid.layers as layers
import ascend_optimizer

ASCEND = True

image = fluid.layers.data(name='image', shape=[3], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
label_index = fluid.layers.data(name='label_index', shape=[2,2], dtype='int32')

fc0 = fluid.layers.fc(image, size=2, act='relu', bias_attr=False, param_attr=fluid.initializer.Constant(value=2.0))

# CLASS_NUM = 10
# fc1 = fluid.layers.fc(fc0, size=CLASS_NUM, bias_attr=False,param_attr=fluid.initializer.Constant(value=2.0))
# layers.Print(fc1)

cross_entropy = fluid.layers.softmax_with_cross_entropy(fc0, label)
layers.Print(cross_entropy)
cost = fluid.layers.reduce_sum(cross_entropy)
layers.Print(cost)
    

optimizer = paddle.optimizer.SGD(learning_rate=0.01)
if ASCEND:
    optimizer = ascend_optimizer.AscendOptimizer(optimizer, fetch_list=[fc0, cross_entropy, cost])
optimizer.minimize(cost, fluid.default_startup_program())

# 打印网络
with open('mnist.start.program', 'w') as fout:
    print(str(fluid.default_startup_program()), file=fout)
with open('mnist.main.program', 'w') as fout:
    print(str(fluid.default_main_program()), file=fout)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

def print_out(out_var):
    data = np.array(out_var, dtype=np.uint8)
    b_arr = data.tobytes()
    arr_2 = np.frombuffer(b_arr, dtype=np.float32)
    return arr_2

for i in range(2):
    out = exe.run(fluid.default_main_program(), 
        feed={"image": np.array([0.2, 0.3, -0.7, 0.1, -0.7, 0.9]).astype("float32").reshape(2,3), 
                "label": np.array([0,1]).astype("int64").reshape(2,1),
                "label_index": np.array([[0,0],[1,1]]).astype("int32").reshape(2,2)},
        fetch_list = [fc0, cross_entropy, cost])

    if ASCEND:
        print("len(out): ", len(out))
        print("type(out[0]): ", type(out[0]))
        for i, out_var in enumerate(out):
            print("%d th output is %s" % (i, str(print_out(out_var))))
print("script done")

# 0 th output is [1.239831  1.0189247]
# 1 th output is [2.2587557]
