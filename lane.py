import numpy as np
import cv2

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
  
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(255, 0, 0), thickness=3):
    while lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                return cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

cap = cv2.VideoCapture('test.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    # convert images to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # to detect yello lane, we need use HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define target range for yello value
    lower_yellow = np.array([20, 100, 100], dtype = "uint8")
    upper_yellow = np.array([30, 255, 255], dtype = "uint8")

    # mask to the original RGB image
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)

    # gaussian blur
    kernel_size = 5
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)

    # cany edge detection
    low_threshold = 5
    high_threshold = 100
    canny_edges = cv2.Canny(gauss_gray,low_threshold,high_threshold)

    # create a region of interest
    max_width = frame.shape[1]
    max_height = frame.shape[0]
    width_delta = int(max_width/20)
    vertices = np.array([[(0, max_height), (max_width/2+140, max_height), (max_width/2 - width_delta/4, max_height/2+170), (max_width/3 - width_delta/4, max_height/2+170)]], np.int32)
    roi = region_of_interest(canny_edges, vertices)

    # transforms these points into lines inside of Hough space
    minLineLength = 1
    maxLineGap = 15000
    lines = hough_lines(roi,1,np.pi/180,50,minLineLength,maxLineGap)

    weight_img = cv2.addWeighted(gray, 0.8, lines, 1., 0.)
    

    # show frame
    cv2.imshow('before',lines)
    cv2.imshow('frame',weight_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
