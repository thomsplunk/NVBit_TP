import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf

tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
b = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
kernel_example_module = tf.load_op_library('/home/tyler/Downloads/nvbit-Linux-aarch64-1.5.5/nvbit_release/tools/nvbitfi/test-apps/rtf/rtf1_kernels.so')
# Run on the GPU
with tf.device('/device:GPU:0'):
    b = kernel_example_module.rtf(a)
#c = tf.matmul(a,b)
print(b)
#print(c)
