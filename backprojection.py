import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('road2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


roi=img[480:500,300:600,:]
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
target = cv2.imread('road2.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

y,x = gray.shape
point=np.array([[300, 440],[400,440],[500,440],[600,440],[250,460],[350,460],[450,460],[550,460]])
picked=np.array([gray[440, 300],gray[440,400],gray[440,500],gray[440,600],gray[460,250],gray[460,350],gray[460,450],gray[460,550]])

###################################################################################################################################
# use numpy instead of opencv
###################################################################################################################################
# M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
  
# R = M/(I+1)
# print M.max(),I.max(),R.dtype
# #cv2.normalize(prob,prob,0,255,cv2.NORM_MINMAX,0)
  
# h,s,v = cv2.split(hsvt)
# B = R[h.ravel(),s.ravel()]
# B = np.minimum(B,1)
# B = B.reshape(hsvt.shape[:2])
  
# disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# cv2.filter2D(B,-1,disc,B)
# B = np.uint8(B)
# cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
# ret,thresh = cv2.threshold(B,50,255,0)

###################################################################################################################################
# use opencv
###################################################################################################################################
# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

# normalize histogram and apply backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
cv2.filter2D(dst,-1,disc,dst)

# threshold and binary AND
# thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
ret,thresh = cv2.threshold(dst,0,255,cv2.THRESH_BINARY)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)
# res = np.vstack((target,res))
res = np.vstack((img,thresh,res))
cv2.imwrite('res.jpg',res)


# plt.figure(0)
# plt.subplot(121)
# plt.imshow(img,cmap = 'gray')
# plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122)
# plt.imshow(gray,cmap = 'gray')
# plt.plot(point[:,0],point[:,1],'r.')
# plt.title('Gray')
# plt.xticks([]), plt.yticks([])

plt.figure(1)
plt.imshow(res)
plt.show()

##################################################################################################################################
##################################################################################################################################
# video test
##################################################################################################################################

# cap = cv2.VideoCapture('test.mp4')

# while(cap.isOpened()):
#     ret, target = cap.read()
#     target = cv2.resize(target, (0,0), fx=0.5, fy=0.5)

#     gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)

#     roi=target[500:520,300:450,:]
#     hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
#     hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

#     roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
#     cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
#     dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

#     disc = cv2.getStructuringElement(cv2.MORPH_CROSS,(11,11))
#     cv2.filter2D(dst,-1,disc,dst)

#     ret,thresh = cv2.threshold(dst,50,255,cv2.THRESH_BINARY)
#     thresh = cv2.merge((thresh,thresh,thresh))
#     res = cv2.bitwise_and(target,thresh)
#     res = np.vstack((target,res))

#     # res = np.vstack((target,thresh,res))

#     cv2.imshow('result',res)
#     # raw_input("Press Enter to continue...")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#     	break


# cap.release()
# cv2.destroyAllWindows()



