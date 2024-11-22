# Hayden Feddock
# 11/22/2024

import numpy as np


# Demosaic the raw image data
def demosaic_img(raw_img: np.ndarray, num_filters: int) -> np.ndarray:
    """
    Demosaic the raw image data from a multispectral camera.
    
    Args:
        raw_img: The raw image data as a ndarray from the camera.
        num_filters: The number of filters used by the multispectral
            camera. It is assumed that the mosaic pattern is a pxp
            square grid such that num_filters = p^2 for some integer p.
        
    Returns:
        The demosaiced image data.
    """

    # Get the height and width of the image
    p = np.sqrt(num_filters)
    if p != int(p):
        raise ValueError("The number of filters must be a perfect square.")
    else:
        p = int(p)
    height = raw_img.shape[0] / p
    width = raw_img.shape[1] / p
    if height != int(height) or width != int(width):
        raise ValueError("The image dimensions must be divisible by the number of filters.")
    else:
        height = int(height)
        width = int(width)

    # Create a reduced image for each filter used
    spectral_cube = np.zeros((height, width, num_filters))

    # Demosaic the image
    for h in range(height):
        for w in range(width):
            region = raw_img[h*p:h*p+p, w*p:w*p+p]
            for i in range(p):
                for j in range(p):
                    spectral_cube[h, w, i*p+j] = region[i, j]

    return spectral_cube



