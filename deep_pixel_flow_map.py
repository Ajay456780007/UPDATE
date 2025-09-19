import cv2
import numpy as np
from keras.applications import ResNet50
from keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load pre-trained ResNet50 (without top)
base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.layers[1].output)


def deep_pixel_flow_map(img):
    """
    1. Pass input image through ResNet50 first (layer 1 output).
    2. Convert extracted feature map to grayscale.
    3. Apply Sobel filter in x and y directions on the feature map.
    4. Compute gradient magnitude.
    5. Resize magnitude to 100x100 and return.

    Parameters:
        img (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Sobel gradient magnitude image resized to 100x100.
    """
    # Resize and convert input image to RGB
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Preprocess for ResNet50 and extract features from layer 1
    x = np.expand_dims(img_rgb.astype(np.float32), axis=0)
    x = preprocess_input(x)
    feature_map = model.predict(x)  # Shape: (1, 112, 112, 64)

    # Remove batch dimension and select channel sum to convert to grayscale-like
    feature_map = np.squeeze(feature_map)  # (112, 112, 64)
    gray_feat = np.sum(feature_map, axis=2)  # (112, 112) sum across channels to grayscale-like

    # Normalize to uint8
    gray_norm = cv2.normalize(gray_feat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Sobel filters
    grad_x = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)

    # Compute gradient magnitude
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize and convert to uint8
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Resize to 100x100
    mag_resized = cv2.resize(mag_norm, (100, 100))

    return mag_resized

# Example usage:
# img = cv2.imread('your_image.jpg')
# flow_map = deep_pixel_flow_map(img)
# import matplotlib.pyplot as plt
# plt.imshow(flow_map, cmap='gray')
# plt.axis('off')
# plt.show()
