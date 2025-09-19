from keras.applications import ResNet101
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
from Sub_Functions.Structural_pattern import StructuralPattern
import numpy as np

# loading the ResNet101 model
model = ResNet101(weights='imagenet', include_top=True)
# creating a new model with only the last layer of the ResNet101 model
model = Model(inputs=model.input, outputs=model.layers[1].output)


def Deep_Structural_Pattern(img):
    # resize the image to 224x224
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    img = cv2.resize(img, (224, 224))
    # predicting the output of the last layer of the ResNet101 model
    input1 = np.expand_dims(img, axis=0)
    # output image
    output = model.predict(input1)
    # squeezing to remove the batch size from the image
    output = np.squeeze(output)
    # creating instance of StructuralPattern class
    SP = StructuralPattern(output)
    # getting the structural pattern
    final_out = SP.get_structural_pattern()
    # returning the final output
    final_out=cv2.resize(final_out, (100,100))
    return final_out  # output shape 224,224
# why an border is created around the resnet image}?

# img = cv2.imread('../Dataset/DB3/rice_leaf_diseases/Leaf smut/DSC_0503.jpg')
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for correct colors
#
# final_out = Deep_Structural_Pattern(img)
#
# # plt.figure(figsize=(10, 5))
# #
# # plt.subplot(1, 2, 1)
# # plt.imshow(img)
# # plt.title('Original Image')
# # plt.axis('off')  # Optional: hide axis
# # plt.subplot(1, 2, 2)
# plt.imshow(final_out, cmap='gray')
# plt.title('Structural Pattern')
# plt.axis('off')  # Optional: hide axis
# plt.show()


