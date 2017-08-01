import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data

import cv2


PATCH_SIZE = 375

# open the camera image
Asphalt = cv2.imread('Asphalt.jpg')
Asphalt = cv2.cvtColor(Asphalt,cv2.COLOR_BGR2GRAY)

# select some patches from Asphalt areas of the image
Asphalt_locations = [(0, 0)]
Asphalt_patches = []
for loc in Asphalt_locations:
    Asphalt_patches.append(Asphalt[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])


# open the camera image
concrete = cv2.imread('concrete.jpg')
concrete = cv2.cvtColor(concrete,cv2.COLOR_BGR2GRAY)

# select some patches from concrete areas of the image
concrete_locations = [(0, 0)]
concrete_patches = []
for loc in concrete_locations:
    concrete_patches.append(concrete[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])


# open the camera image
gravel= cv2.imread('gravel.jpg')
gravel = cv2.cvtColor(gravel,cv2.COLOR_BGR2GRAY)

# select some patches from gravel areas of the image
gravel_locations = [(0, 0)]
gravel_patches = []
for loc in gravel_locations:
    gravel_patches.append(gravel[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])


# open the camera image
sand= cv2.imread('sand.jpg')
sand = cv2.cvtColor(sand,cv2.COLOR_BGR2GRAY)

# select some patches from sand areas of the image
sand_locations = [(0, 0)]
sand_patches = []
for loc in sand_locations:
    sand_patches.append(sand[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])


# compute some GLCM properties each patch
xs = []
ys = []
for patch in (Asphalt_patches + concrete_patches + gravel_patches + sand_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
# ax = fig.add_subplot(3, 2, 1)
# ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
#           vmin=0, vmax=255)
# for (y, x) in Asphalt_locations:
#     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
# for (y, x) in concrete_locations:
#     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
# for (y, x) in gravel_locations:
#     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'rs')
# for (y, x) in sand_locations:
#     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'ys')
# ax.set_xlabel('Original Image')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 1, 1)
ax.plot(xs[0], ys[0], 'go',
        label='Asphalt')
ax.plot(xs[1], ys[1], 'bo',
        label='concrete')
ax.plot(xs[2], ys[2], 'ro',
        label='gravel')
ax.plot(xs[3], ys[3], 'yo',
        label='sand')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i,patch in enumerate(Asphalt_patches + concrete_patches + gravel_patches + sand_patches):
  ax = fig.add_subplot(3, 4, i+5)
  ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',vmin=0, vmax=255)
# ax.set_xlabel('Asphalt_patches')
# ax = fig.add_subplot(3, 4, 6)
# ax.imshow(concrete_patches, cmap=plt.cm.gray, interpolation='nearest',vmin=0, vmax=255)
# ax.set_xlabel('concrete_patches')
# ax = fig.add_subplot(3, 4, 7)
# ax.imshow(gravel_patches, cmap=plt.cm.gray, interpolation='nearest',vmin=0, vmax=255)
# ax.set_xlabel('gravel_patches')
# ax = fig.add_subplot(3, 4, 8)
# ax.imshow(sand_patches, cmap=plt.cm.gray, interpolation='nearest',vmin=0, vmax=255)
# ax.set_xlabel('sand_patches')

# for i, patch in enumerate(sky_patches):
#     ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
#     ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
#               vmin=0, vmax=255)
#     ax.set_xlabel('Sky %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()