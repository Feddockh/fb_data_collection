import numpy as np
from ximea import xiapi
import matplotlib.pyplot as plt
from multispectral_imaging import demosaic_img

cam = xiapi.Camera()
cam.open_device_by_SN('32680454')

# Take a photo with the camera
cam.set_exposure(100000)
img = xiapi.Image()
cam.start_acquisition()
cam.get_image(img)

# Get the raw data from the camera image
data_raw = img.get_image_data_raw()
with open('data/captured_data_raw.bin', 'wb') as f:
    f.write(data_raw)
data = list(data_raw)

# Stop the camera
cam.stop_acquisition()
cam.close_device()

# Print the image data and metadata
np_data = np.array(data)
with open('data/captured_image_raw.npy', 'wb') as f:
    np.save(f, np_data)
np_data = np_data.reshape((img.height, img.width))
print('Image shape: ' + str(np_data.shape))

# Display the image
plt.imshow(np_data, cmap='gray')
plt.title('Captured Image')
plt.show()

# # Demosaic the image
# spectral_cube = demosaic_img(np_data, 16)
# print('Spectral cube shape: ' + str(spectral_cube.shape))

# # Display the spectral cube images
# fig, axes = plt.subplots(4, 4, figsize=(15, 15))
# fig.suptitle('Spectral Cube Layers')

# for i in range(4):
#     for j in range(4):
#         ax = axes[i, j]
#         ax.imshow(spectral_cube[:, :, i * 4 + j], cmap='gray')
#         ax.set_title(f'Layer {i * 4 + j + 1}')
#         ax.axis('off')

# plt.tight_layout()
# plt.show()


















