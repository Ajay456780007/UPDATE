import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops


def glcm_statistical_features(img):
    # Read image
    # img = cv2.imread(image_path)

    # Convert to grayscale
    # img_uint8 = np.uint8(img * 255)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.uint8(gray)
    # Compute GLCM
    glcm = graycomatrix(gray,
                        distances=[1],
                        angles=[0],
                        levels=256,
                        symmetric=True,
                        normed=True)

    # Extract features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Print feature values
    features = {
        "Contrast": contrast,
        "Correlation": correlation,
        "Energy": energy,
        "Homogeneity": homogeneity
    }

    print("GLCM Statistical Features:")
    for k, v in features.items():
        print(f"{k}: {v:.4f}")

    # Normalize the GLCM for display
    glcm_image = glcm[:, :, 0, 0]  # Take the 2D slice from 4D matrix
    glcm_image_normalized = (glcm_image / glcm_image.max()) * 255
    glcm_image_normalized = glcm_image_normalized.astype(np.uint8)
    glcm_image_normalized = cv2.resize(glcm_image_normalized, (100,100))

    # # Show images and GLCM
    # plt.figure(figsize=(15, 5))
    #
    # plt.subplot(1, 3, 1)
    # plt.title("Original Grayscale Image")
    # plt.imshow(gray, cmap='gray')
    # plt.axis("off")
    #
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # #
    # plt.subplot(1, 2, 2)
    # plt.title("GLCM Matrix (Visualized)")
    # plt.imshow(glcm_image_normalized, cmap='gray')
    # plt.axis("off")
    #
    # plt.tight_layout()
    # plt.show()

    return glcm_image_normalized

# Example usage:
# features, glcm_visual = glcm_statistical_features("../Dataset/DB1/archive/Bacterialblight/BACTERAILBLIGHT3_003.jpg")
