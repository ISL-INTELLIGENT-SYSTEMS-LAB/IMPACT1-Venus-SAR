import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

path = r'C:\Users\forth\Desktop\MaskFormer\data\preprocessed_labels\0_-1_BL.png'

img = cv2.imread(path)

plt.imshow(img)
#plt.show()

print(img.shape)

imgnp = np.array(img) + 1
print(np.unique(imgnp))

ann = imgnp[:, :, 0]

plt.imshow(ann)
plt.show()