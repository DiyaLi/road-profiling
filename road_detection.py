import matplotlib.pyplot as plt
import numpy as np
import cv2

# dege detection

def draw_lines(img, lines, color=(255, 0, 0), thickness=3):
    while lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                return cv2.line(img, (x1, y1), (x2, y2), color, thickness)



img = cv2.imread('road.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
im_y, im_x =gray.shape
corp_im = gray[im_y/2:im_y,im_x/4:3*im_x/4]

# cany edge detection
corp_gauss_gray=cv2.GaussianBlur(corp_im, (9,5), 0)
low_threshold = 5
high_threshold = 100
canny_edges = cv2.Canny(corp_gauss_gray,low_threshold,high_threshold)


lines = cv2.HoughLines(canny_edges,1,np.pi/180,200)

# draw lines on image
while lines is not None:
	for rho,theta in lines[0]:
    	a = np.cos(theta)
   		b = np.sin(theta)
    	x0 = a*rho
    	y0 = b*rho
    	x1 = int(x0 + 1000*(-b))
    	y1 = int(y0 + 1000*(a))
    	x2 = int(x0 - 1000*(-b))
    	y2 = int(y0 - 1000*(a))
    	cv2.line(corp_im,(x1,y1),(x2,y2),(0,0,255),2)


plt.subplot(311),plt.imshow(corp_gauss_gray,cmap = 'gray')
plt.title('corp_gauss_gray'), plt.xticks([]), plt.yticks([])
plt.subplot(312),plt.imshow(canny_edges,cmap = 'gray')
plt.title('canny_edges'), plt.xticks([]), plt.yticks([])
plt.subplot(313),plt.imshow(corp_im,cmap = 'gray')
plt.title('gray'), plt.xticks([]), plt.yticks([])
plt.show()

cap.release()
cv2.destroyAllWindows()



# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# plt.imshow(thresh,cmap = 'gray')
# plt.xticks([]), plt.yticks([])
# plt.show()
