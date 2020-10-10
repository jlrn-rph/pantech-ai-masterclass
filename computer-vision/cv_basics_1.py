# BASIC OPERATIONS & SYNTAX IN OPENCV

# import libraries
import cv2
import imutils 

# read image
img = cv2.imread('img/lena.jpg')

# resize image
resizeImg = imutils.resize(img, width = 20)

# blur image
gaussianImg = cv2.GaussianBlur(img, (21,21),0)

# threshold image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresImg = cv2.threshold(grayImg, 120, 255, cv2.THRESH_BINARY)

# save image copy
cv2.imwrite('img/lena-resize.jpg', img)
cv2.imwrite('img/lena-gaussian.jpg', gaussianImg)
cv2.imwrite('img/lena-threshold.jpg', thresImg)

# show image
cv2.imshow('lena', img)
cv2.imshow('lena-resize', resizeImg)
cv2.imshow('lena-gaussian', gaussianImg)
cv2.imshow('lena-gaussian', thresImg)

# close windows
cv2.waitKey(0)
cv2.destroyAllWindows()