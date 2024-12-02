import cv2
import numpy as np
import torch
from PIL import Image
from retinaface.pre_trained_models import get_model
from skimage.morphology import dilation, square
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.fftpack import fft2, fftshift
import logging
from typing import List, Tuple, Optional, Dict, Any
import os

# Initialize face detector
face_detector = get_model("resnet50_2020-07-20", max_size=2048, device="cuda" if torch.cuda.is_available() else "cpu")
face_detector.eval()

FILTERS = [
    ("resolution", "Check minimum resolution"),
    ("grayscale", "Check for grayscale images"),
    ("focus", "Check image focus"),
    ("noise", "Check image noise"),
    ("face", "Check face detection"),
    ("border", "Check for borders"),
    ("jpeg_artifacts", "Check JPEG artifacts")
]

def validate_image(image: Any) -> Optional[np.ndarray]:
    """Validate and convert the image to a numpy array."""
    if image is None:
        logging.warning("Received None as image input")
        return None
    
    if isinstance(image, np.ndarray):
        if image.ndim not in (2, 3):
            logging.warning(f"Invalid image dimensions: {image.shape}")
            return None
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, str):
        try:
            return np.array(Image.open(image))
        except Exception as e:
            logging.error(f"Error opening image file: {e}")
            return None
    else:
        logging.warning(f"Unsupported image type: {type(image)}")
        return None

def check_grayscale(image):
    if len(image.shape) == 2:
        return True
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    
    if np.array_equal(hist_r, hist_g) and np.array_equal(hist_r, hist_b):
        return True
    
    hist_3d = cv2.calcHist([image_rgb], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    diagonal_sum = np.sum([hist_3d[i, i, i] for i in range(256)])
    total_sum = np.sum(hist_3d)
    
    if diagonal_sum / total_sum > 0.9:
        return True
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = image_hsv[:, :, 1]
    mean_saturation = np.mean(saturation)
    std_saturation = np.std(saturation)
    
    if mean_saturation < 10 and std_saturation < 5:
        return True
    
    return False

def tenengrad_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.square(gx) + np.square(gy))

def estimate_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    std_dev = np.std(gray)
    entropy = -np.sum(np.histogram(gray, bins=256, density=True)[0] * 
                      np.log2(np.histogram(gray, bins=256, density=True)[0] + 1e-7))
    
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    
    n_points = 8
    radius = 1
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    mean_magnitude = np.mean(magnitude_spectrum)
    
    noise_level = (
        0.2 * std_dev +
        0.1 * entropy +
        0.2 * edge_density +
        0.1 * contrast +
        0.1 * (1 - homogeneity) +
        0.1 * mean_magnitude +
        0.2 * np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
    )
    
    normalized_noise_level = 100 * (noise_level - 0) / (100 - 0)
    
    return np.clip(normalized_noise_level, 0, 100)

def detect_border(img, threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top_edge = gray[0, :]
    bottom_edge = gray[-1, :]
    left_edge = gray[:, 0]
    right_edge = gray[:, -1]
    # Check top and bottom edges
    top_edge = gray[0, :]
    bottom_edge = gray[-1, :]
    
    # Check left and right edges
    left_edge = gray[:, 0]
    right_edge = gray[:, -1]
    
    # Function to check if edge has consistent color
    def is_consistent_edge(edge):
        return np.all(np.abs(edge - np.mean(edge)) < threshold)
    
    # Check if all edges are consistent
    if (is_consistent_edge(top_edge) and
        is_consistent_edge(bottom_edge) and
        is_consistent_edge(left_edge) and
        is_consistent_edge(right_edge)):
        
        # Additional check: ensure border is different from image content
        inner_region = gray[10:-10, 10:-10]
        border_color = np.mean([top_edge, bottom_edge, left_edge, right_edge])
        if abs(np.mean(inner_region) - border_color) > threshold:
            return True
    
    return False

def detect_jpeg_artifacts(image, threshold=85):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(edge_h**2 + edge_v**2)
    block_artifact_score = np.mean(edge_mag[::8, :]) + np.mean(edge_mag[:, ::8])
    return block_artifact_score


def process_single_image(image):
    img_array = validate_image(image)
    if img_array is None:
        return {
            "error": "Invalid image input",
            "resolution": ("N/A", False),
            "grayscale": ("N/A", False),
            "focus": ("N/A", False),
            "noise": ("N/A", False),
            "face": ("N/A", False),
            "border": ("N/A", False),
            "jpeg_artifacts": ("N/A", False)
        }

    try:
        height, width = img_array.shape[:2]
        resolution = f"{width}x{height}"
        resolution_pass = width >= 1024 and height >= 1024

        is_grayscale = check_grayscale(img_array)
        
        focus_measure = tenengrad_variance(img_array)
        focus_pass = focus_measure > 2000

        noise_level = estimate_noise(img_array)
        noise_pass = noise_level < 80

        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        faces = face_detector.predict_jsons(img_bgr)
        face_pass = len(faces) == 1 if faces else False

        has_border = detect_border(img_array)
        
        jpeg_score = detect_jpeg_artifacts(img_array)
        jpeg_pass = jpeg_score < 70

        return {
            "resolution": (resolution, resolution_pass),
            "grayscale": ("Color" if not is_grayscale else "Grayscale", not is_grayscale),
            "focus": (f"{focus_measure:.2f}", focus_pass),
            "noise": (f"{noise_level:.2f}", noise_pass),
            "face": (f"{len(faces)} face(s)", face_pass),
            "border": ("Border" if has_border else "No border", not has_border),
            "jpeg_artifacts": (f"{jpeg_score:.2f}", jpeg_pass)
        }
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return {
            "error": f"Error processing image: {str(e)}",
            "resolution": ("N/A", False),
            "grayscale": ("N/A", False),
            "focus": ("N/A", False),
            "noise": ("N/A", False),
            "face": ("N/A", False),
            "border": ("N/A", False),
            "jpeg_artifacts": ("N/A", False)
        }
