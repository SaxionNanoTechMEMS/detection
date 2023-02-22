import cv2
import numpy as np
import os
# Load the image
input1=str(input())
path=os.getcwd()
link=os.path.join(path,'data',input1)

image = cv2.imread(link)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the image to create a binary image
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Apply morphological operations to the binary image to remove the small noise and fill the holes in the object
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(binary,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 2)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=3)

# Find the contours in the image
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on a blank image and fill them with white color
mask = np.zeros_like(image)
cv2.drawContours(mask, contours, -1, (255,255,255), -1)

# Apply the mask to the original image to remove the background
result = cv2.bitwise_and(image, mask)

# Save the output image
cv2.imwrite('RF'+input1, result)
