import cv2
import numpy as np


def ROI_Extraction(img_):
    # img_ = cv2.resize(img_, (4, 400))
    # --------- region of Interest Segmentation ---------------
    img = cv2.resize(img_, (224, 224))
    hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)  # convert to hue saturation value
    # range of green color in HSV
    l_g = np.array([30, 140, 40])  # lower bound
    u_g = np.array([90, 255, 100])  # upper bound
    # mask
    mask = cv2.inRange(hsv, l_g, u_g)
    # Apply morphological operations to remove small noise and fill gaps in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Adjust kernel size
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If contours are found, proceed to extract the ROI
    if contours:
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        # Get bounding box coordinates of the largest contour
        # x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # Extract the region of interest (ROI)
        roi = img_[x: x + w, y: y + h]
        if w == 0 or h == 0 or x == 0 or y == 0:
            roi = img_
        else:
            roi = cv2.resize(roi, (224, 224))
    else:
        roi = img_
        roi = cv2.resize(roi, (224, 224))
    return img

# img=cv2.imread('../Dataset/DB1/archive/Bacterialblight/BACTERAILBLIGHT3_009.jpg')
# roi=ROI_Extraction(img)
# cv2.imshow('ROI',roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
