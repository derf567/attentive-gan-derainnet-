from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
from skimage.transform import resize
import numpy as np

img_output = imread('Output.png')
img_rain = imread('rain_26.png')
img_clean = imread('clean_22.png')

# Resize all images to match shape and channels
target_shape = img_clean.shape
img_output_resized = resize(img_output, target_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)
img_rain_resized = resize(img_rain, target_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)

# Set data_range according to dtype and value range
data_range = 255

# Output vs Groundtruth
psnr_output = peak_signal_noise_ratio(img_clean, img_output_resized, data_range=data_range)
ssim_output = structural_similarity(img_clean, img_output_resized, data_range=data_range, channel_axis=-1)

# Rain (input) vs Groundtruth
psnr_rain = peak_signal_noise_ratio(img_clean, img_rain_resized, data_range=data_range)
ssim_rain = structural_similarity(img_clean, img_rain_resized, data_range=data_range, channel_axis=-1)

print('Output vs Groundtruth: PSNR =', psnr_output, ', SSIM =', ssim_output)
print('Rain (input) vs Groundtruth: PSNR =', psnr_rain, ', SSIM =', ssim_rain)
