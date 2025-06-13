import tensorflow as tf

print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU available:", tf.test.is_gpu_available())
