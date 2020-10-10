# import libraries
import cv2

# read image
img = cv2.imread('img/lena.jpg')

# convert to grayscale image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# get properties
print('Size:', img.size)
print('Shape:', img.shape)
print('Data Type:', img.dtype)

# show image
cv2.imshow("lena-original", img)
cv2.imshow("lena-gray", grayImg)

# save image copy
cv2.imwrite('img/lena-copy.jpg', img)

# save grayscale image copy
cv2.imwrite('img/lena-gray.jpg', img)

# closing windows
cv2.waitKey(0)
cv2.destroyAllWindows()
