import numpy as np
from keras.applications import ResNet152
from keras import Model
import cv2

model = ResNet152(weights="imagenet", include_top=True)
model = Model(inputs=model.input, outputs=model.layers[1].output)


def Resnet151(img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)
    output = np.squeeze(output)
    final_out = output[:, :, 1]
    final_out = cv2.resize(final_out, (224, 224))
    return final_out


# import matplotlib.pyplot as plt
# img = cv2.imread('../Dataset/DB3/rice_leaf_diseases/Leaf smut/DSC_0503.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for correct colors
#
# final_out = Resnet151(img)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.title('Original Image')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(final_out, cmap='gray')
# plt.box(False)
# plt.title('Resnet151')
# plt.axis('off')
# plt.show()

