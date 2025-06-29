import matplotlib.pyplot as plt
from PIL import Image
import os

img = Image.open("multicenter/external/Dataset004_SierraLeone/imagesTr/003_0000.png")
mask = Image.open("multicenter/external/Dataset004_SierraLeone/labelsTr/filled_003_gt_003.png")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Image")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Mask")
plt.show()
