# Hayden Feddock
# 11/24/2024

import numpy as np
import cv2
from typing import List

def get_camera_intrinsics(image: np.ndarray, chessboard_size: tuple) -> tuple:
    """
    Get the camera intrinsics from a calibration image of a chessboard.
    
    Args:
        image: The calibration image as a ndarray.
        chessboard_size: The size of the chessboard as a tuple (rows, cols).
        
    Returns:
        The camera matrix and distortion coefficients.
    """
    
    # Find the chessboard corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    # Get the camera intrinsics
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [np.zeros(corners.shape).astype(np.float32)], [corners], gray.shape[::-1], None, None
    )
    
    return camera_matrix, dist_coeffs

def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort an image using the camera matrix and distortion coefficients.
    
    Args:
        image: The image to undistort as a ndarray.
        camera_matrix: The camera matrix as a ndarray.
        dist_coeffs: The distortion coefficients as a ndarray.
        
    Returns:
        The undistorted image.
    """
    
    return cv2.undistort(image, camera_matrix, dist_coeffs)

def calibrate_camera(calibration_images: List[np.ndarray], chessboard_size: tuple) -> tuple:
    """
    Calibrate the camera using a set of calibration images.
    
    Args:
        calibration_images: A list of calibration images as ndarrays.
        chessboard_size: The size of the chessboard as a tuple (rows, cols).
        
    Returns:
        The camera matrix and distortion coefficients.
    """
    
    # Get the camera intrinsics for each calibration image
    camera_matrices = []
    dist_coeffs = []
    for image in calibration_images:
        camera_matrix, dist_coeff = get_camera_intrinsics(image, chessboard_size)
        camera_matrices.append(camera_matrix)
        dist_coeffs.append(dist_coeff)
    
    # Average the camera matrices and distortion coefficients
    camera_matrix = np.mean(camera_matrices, axis=0)
    dist_coeffs = np.mean(dist_coeffs, axis=0)
    
    return camera_matrix, dist_coeffs

