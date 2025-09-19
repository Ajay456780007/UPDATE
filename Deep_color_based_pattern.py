import numpy as np
import cv2
from keras.applications import ResNet50
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Step 0: Load ResNet50 and set up feature extractor for layer 1
base_model = ResNet50(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.layers[1].output)


def deep_color_based_pattern(img):
    # =========================================================
    # 1. Preprocess image: ensure RGB and resize for ResNet
    # =========================================================
    if img.ndim == 2 or img.shape[2] == 1:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resnet = cv2.resize(img_rgb, (224, 224))
    x = np.expand_dims(img_resnet.astype(np.float32), axis=0)
    x = preprocess_input(x)

    # =========================================================
    # 2. Extract features from ResNet50 (layer 1)
    # =========================================================
    feature_map = model.predict(x)[0]  # feature_map shape: (112,112,64) for layer 1

    # Use a representative channel for further processing
    channel_idx = 1
    resnet_feature = feature_map[:, :, channel_idx]
    # Normalize to [0,255] and convert back to uint8 image
    resnet_image = cv2.normalize(resnet_feature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Resize back to original or keep at (112,112) for speed
    resnet_image_3ch = cv2.merge([resnet_image] * 3)

    # =========================================================
    # 3. Apply Fast Adaptive Median Filter
    # =========================================================
    def fast_adaptive_median(img):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        k = 3  # kernel size for median blur
        filtered = cv2.medianBlur(img, k)
        return filtered

    smoothed = fast_adaptive_median(resnet_image_3ch)

    # =========================================================
    # 4. Weighted sum transformation (Eq. 1)
    # =========================================================
    R, G, B = cv2.split(smoothed)
    Tpix = 2 * R.astype(np.float32) + 3 * G.astype(np.float32) + 4 * B.astype(np.float32)

    # =========================================================
    # 5. Directional color differences (vectorized)
    # =========================================================
    diff0 = np.abs(np.roll(Tpix, -1, axis=1) - np.roll(Tpix, 1, axis=1))  # horizontal
    diff90 = np.abs(np.roll(Tpix, -1, axis=0) - np.roll(Tpix, 1, axis=0))  # vertical
    diff135 = np.abs(
        np.roll(np.roll(Tpix, -1, axis=0), -1, axis=1) - np.roll(np.roll(Tpix, 1, axis=0), 1, axis=1))  # 135°
    diff45 = np.abs(
        np.roll(np.roll(Tpix, 1, axis=0), -1, axis=1) - np.roll(np.roll(Tpix, -1, axis=0), 1, axis=1))  # 45°

    # Zero out invalid border values
    diff0[:, 0] = diff0[:, -1] = 0
    diff90[0, :] = diff90[-1, :] = 0
    diff135[0, :] = diff135[-1, :] = 0
    diff45[0, :] = diff45[-1, :] = 0

    # =========================================================
    # 6. Max directional difference and threshold
    # =========================================================
    max_diff = np.maximum.reduce([diff0, diff90, diff135, diff45])
    T = 1.2 * np.mean(max_diff)
    edge_map = (max_diff >= T).astype(np.uint8) * 255

    # =========================================================
    # 7. Fast Morphological Thinning – Two Masks
    # =========================================================
    def thinning_two_masks(edge_img):
        mask_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        mask_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        thin_h = cv2.erode(edge_img, mask_h)
        thin_v = cv2.erode(edge_img, mask_v)
        return cv2.max(thin_h, thin_v)

    thinned_edges = thinning_two_masks(edge_map)

    # =========================================================
    # 7b. Invert for white background and lighten edges
    # =========================================================
    edges_inverted = cv2.bitwise_not(thinned_edges)
    edges_light = cv2.addWeighted(edges_inverted, 0.8, 255 * np.ones_like(edges_inverted), 0.2, 0)

    # =========================================================
    # 8. Final edge pattern — resize for display
    # =========================================================
    edges_final = cv2.resize(edges_light, (100,100))
    return edges_final

# Usage Example
# img = cv2.imread('your_image.jpg')
# result = deep_color_based_pattern(img)
# plt.imshow(result, cmap='gray')
# plt.axis('off')
# plt.show()
