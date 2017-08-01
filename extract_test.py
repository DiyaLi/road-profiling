import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('road.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

height,width=gray.shape
template = gray[height*3/4:height,width/2-80:width*3/4-100]
# template = gray[100:300,100:300]
edges = cv2.Canny(template,100,200)

k_xr = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
sobel_xr = cv2.filter2D(template,-1,k_xr)

ex_sharping = np.array([[1,1,1],[1,-7,1],[1,1,1]])
sharp = cv2.filter2D(template,-1,ex_sharping)

plt.subplot(121)
plt.imshow(sharp,cmap = 'gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(sobel_xr,cmap = 'gray')
plt.title('template')
plt.xticks([]), plt.yticks([])
plt.show()



