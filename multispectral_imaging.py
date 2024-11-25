# Hayden Feddock
# 11/22/2024

import numpy as np
from typing import List
from scipy.interpolate import make_interp_spline

def get_spectral_cube(raw_img: np.ndarray, num_filters: int) -> np.ndarray:
    """
    Demosaic the raw image data from a multispectral camera and return
    the spectral cube.
    
    Args:
        raw_img: The raw image data as a ndarray from the camera.
        num_filters: The number of filters used by the multispectral
            camera. It is assumed that the mosaic pattern is a pxp
            square grid such that num_filters = p^2 for some integer p.
        
    Returns:
        The demosaiced spectral cube.
    """

    # Check that the number of filters is a perfect square
    p = np.sqrt(num_filters)
    if p != int(p):
        raise ValueError("The number of filters must be a perfect square.")
    p = p.astype(int)

    # Get the dimensions of the multispectral cube
    new_height = raw_img.shape[0] // p
    new_width = raw_img.shape[1] // p
    spectral_cube = np.zeros((new_height, new_width, num_filters))

    # Demosaic the image
    for i in range(num_filters):
        row = i // p
        col = i % p
        res = raw_img[row::p, col::p]

        # Fit the result to the new height and width
        res = res[:new_height, :new_width]
        spectral_cube[:, :, i] = res

    return spectral_cube

def display_spectral_cube(spectral_cube: np.ndarray, colormap: str = 'gray', \
                           labels: List[str] = None) -> None:
    """
    Display the images at each layer of the spectral cube.
    
    Args:
        spectral_cube: The demosaiced spectral cube as a ndarray.
        colormap: The colormap to use for displaying the images
    """

    # Display the image at each layer of the spectral cube
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle('Spectral Cube Layers')
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.imshow(spectral_cube[:, :, i * 4 + j], cmap=colormap)
            if labels is not None:
                ax.set_title(labels[i * 4 + j])
            else:
                ax.set_title(f'Layer {i * 4 + j + 1}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def display_spectrum(spectral_cube: np.ndarray, x: int, y: int, \
                     wavelengths: List[int]) -> None:
    """
    Display the spectrum at a given pixel location in the spectral cube.
    
    Args:
        spectral_cube: The demosaiced spectral cube as a ndarray.
        x: The x-coordinate of the pixel.
        y: The y-coordinate of the pixel.
        wavelengths: The wavelengths corresponding to each layer of the
            spectral cube.
    """

    # Get the spectrum at the given pixel location
    spectrum = spectral_cube[y, x, :]

    if wavelengths is None:
        wavelengths = np.arange(0, spectral_cube.shape[2])
    else:
        if len(wavelengths) != spectral_cube.shape[2]:
            raise ValueError("Number of wavelengths must match the number of \
                             layers in the spectral cube.")

    # Plot the spectrum with a smooth curve
    # Create a smooth curve
    wavelengths_new = np.linspace(wavelengths.min(), wavelengths.max(), 300)
    spline = make_interp_spline(wavelengths, spectrum, k=3)
    spectrum_smooth = spline(wavelengths_new)

    # Plot the smooth spectrum
    plt.plot(wavelengths_new, spectrum_smooth)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title(f'Spectrum at Pixel ({x}, {y})')
    plt.show()

if __name__ == "__main__":
        
    # Import the numpy array from the data folder
    np_data = np.load('data/captured_image_raw.npy')

    # Reshape the image data
    np_data = np_data.reshape((1088, 2048))

    # Display the image
    import matplotlib.pyplot as plt
    plt.imshow(np_data, cmap='gray')
    plt.title('Captured Image')
    plt.show()

    # Demosaic the image
    spectral_cube = get_spectral_cube(np_data, 25)

    # Display the spectral cube images
    display_spectral_cube(spectral_cube)

    # Display the spectrum at a given pixel location
    display_spectrum(spectral_cube, 100, 100, None)

