import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import os
import ascend_optimizer
import numpy as np

x1 = fluid.data(name="x1", shape=[2,1], dtype='float32')
x2 = fluid.data(name="x2", shape=[2,2], dtype='float32')
reduce_sum = layers.reduce_sum(x2, dim=0, keep_dim=True)
y = layers.matmul(x1, reduce_sum)
cost = layers.reduce_sum(y)

optimizer = paddle.optimizer.SGD(learning_rate=0.01)
optimizer = ascend_optimizer.AscendOptimizer(optimizer, fetch_list=[reduce_sum, y, cost])
optimizer.minimize(cost, fluid.default_startup_program())

with open("start.txt","w") as fout:
    fout.write(str(fluid.default_startup_program()))
with open("main.txt","w") as fout:
    fout.write(str(fluid.default_main_program()))

exe = fluid.Executor(fluid.CPUPlace())
# exe.run(fluid.default_startup_program())


def print_out(out_var):
    data = np.array(out_var, dtype=np.uint8)
    b_arr = data.tobytes()
    arr_2 = np.frombuffer(b_arr, dtype=np.float32)
    return arr_2

for i in range(2):
    out = exe.run(fluid.default_main_program(), 
        feed={"x1": np.array([1,2]).astype("float32").reshape(2,1), "x2": np.array([3,4,5,6]).astype("float32").reshape(2,2)},
        fetch_list = [reduce_sum, y, cost])

    for i, out_var in enumerate(out):
        print("%d th output is %s" % (i, str(print_out(out_var))))
print("script done")


# 0 th output is [ 8. 10.]
# 1 th output is [ 8. 10. 16. 20.]
# 2 th output is [24. 30.]