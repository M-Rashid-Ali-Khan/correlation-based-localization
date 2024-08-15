import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.signal import correlate2d
from skimage.transform import warp_polar

# Load images (assuming grayscale images)
image1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)

# Preprocess images (e.g., normalization, filtering if necessary)
image1 = image1.astype(np.float32) / 255.0
image2 = image2.astype(np.float32) / 255.0

# Calculate the Fourier transform and its magnitude
def magnitude_spectrum(image):
    f_transform = fft2(image)
    f_shifted = fftshift(f_transform)
    magnitude = np.abs(f_shifted)
    return magnitude

# Convert to polar coordinates
def polar_transform(image):
    magnitude = magnitude_spectrum(image)
    polar_img = warp_polar(magnitude, scaling='log')
    return polar_img

# Estimate rotation by cross-correlation in the polar domain
polar_image1 = polar_transform(image1)
polar_image2 = polar_transform(image2)

# Cross-correlate the polar-transformed magnitude spectra
correlation = correlate2d(polar_image1, polar_image2)
rotation_index = np.unravel_index(np.argmax(correlation), correlation.shape)
rotation_angle = rotation_index[1] * (360 / polar_image1.shape[1])

# Rotate image1 by the estimated angle to align with image2
center = tuple(np.array(image1.shape) / 2)
rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
aligned_image1 = cv2.warpAffine(image1, rotation_matrix, image1.shape[::-1])

# Estimate translation by cross-correlation
translation_correlation = correlate2d(aligned_image1, image2)
translation_index = np.unravel_index(np.argmax(translation_correlation), translation_correlation.shape)
translation_vector = np.array(translation_index) - np.array(aligned_image1.shape) / 2

