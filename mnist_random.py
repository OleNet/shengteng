# UTF-8 -*-
from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import paddle
import paddle.fluid.layers as layers
import ascend_optimizer

ASCEND = True
#ASCEND = False

batch_size = 3
cls_num = 2
image_size = 2

image = fluid.layers.data(name='image', shape=[image_size], dtype='float32', stop_gradient=False)
index = fluid.layers.data(name="index", shape=[], dtype="int32", stop_gradient=False)
label = fluid.layers.data(name='label', shape=[1], dtype='int64', stop_gradient=False)


image.stop_gradient=True
if ASCEND == False:
    image = fluid.layers.Print(image, message="image_layer_norm")

fc0 = fluid.layers.fc(image, size=3, act=None, bias_attr=False, param_attr=fluid.initializer.Constant(value=2.0))
fc1 = fluid.layers.fc(fc0, size=3, act=None, bias_attr=False, param_attr=fluid.initializer.Constant(value=2.0)) #, param_attr=fluid.initializer.TruncatedNormal(loc=0.0, scale=0.02, seed=0))

x = layers.cast(layers.equal(fc0, fc0), dtype='float32')

fc2 = fc1 * x


if ASCEND == False:
    fc0 = layers.Print(fc0, message="fc0")
    fc1 = layers.Print(fc1, message="fc1")

rr = fluid.layers.uniform_random([2,3])
fc2 = fc2 + rr

cross_entropy = fluid.layers.softmax_with_cross_entropy(fc2, label)

if ASCEND == False: cross_entropy = layers.Print(cross_entropy, message="cross_entropy")

expand = fluid.layers.expand(fc0, [1, 2])


cost = fluid.layers.reduce_sum(cross_entropy) + 10
if ASCEND == False: cost = layers.Print(cost, message="cost")

optimizer = paddle.optimizer.SGD(learning_rate=0.01)

fetch_list = [rr]
if ASCEND:
    optimizer = ascend_optimizer.AscendOptimizer(optimizer, fetch_list=fetch_list)

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
    print(arr_2.shape)
    return arr_2


np.random.seed(1)
for i in range(1):
    out = exe.run(fluid.default_main_program(), 
        feed={"image": np.array([0.2, 0.3, 0.7, 0.1, 0.7, 0.9]).astype("float32").reshape(batch_size, image_size),
        "index": np.array([0,1,2]).astype("int32").reshape(3),
        "label": np.array([0,1,0]).astype("int64").reshape(3,1),
        },
        fetch_list = fetch_list)
    if not ASCEND:
        for i in out:
            print(i)

    if ASCEND:
        print("len(out): ", len(out))
        print("type(out[0]): ", type(out[0]))
        for i, out_var in enumerate(out):
            print("%d th output is %s" % (i, str(print_out(out_var))))
print("script done")


