import numpy as np

x = np.random.randn(10, 32, 32, 3) # 10 images, each with 32x32 pixels and 3 channels (RGB)

i = 10 # row index
j = 10 # column index
k = 5 # patch size

patch = x[:, i:i+k, j:j+k, :]

print(patch.shape) # (10, 5, 5, 3)
