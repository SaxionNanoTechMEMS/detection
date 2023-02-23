import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
Installation
1.Python : 3.10.9
2.OpenCV : 3.7.0 !pip install opencv-python
3.Matplotlib !pip install matplotlib.pyplot
4.Numpy !pip isntall numpy

"""

"""
The script has three main steps:

Splitting the images into 3x3 regions: 
This step divides the images into 9 regions of equal size. 
This will allow us to analyze each region independently.

Calculating histograms for each region: 
A histogram is a graph that shows the frequency distribution of pixel values in an image. 
In this step, we calculate a histogram for each region of the two images we want to compare. 
This will give us a representation of the pixel values in each region.

Comparing histograms: 
Finally, we compare the histograms of the corresponding regions in the two images. 
This will allow us to determine the similarity or dissimilarity between the two images in each region. 
We can then visualize the differences by plotting the histograms of each region for each image and the difference between them.
"""

# Read images and convert them to grayscale
# Reseize the images so i can do operations later on
img1 = cv.imread('test.png', 0)
img2 = cv.imread('model.png', 0)
img1 = cv.resize(img1, (250, 250))
img2 = cv.resize(img2, (250, 250))

# Draw a 3X3 rechtangle on the images
h, w = img1.shape[:2]
dh = h // 3
dw = w // 3
for i in range(3):
    for j in range(3):
        cv.rectangle(img1, (j*dw, i*dh), ((j+1)*dw, (i+1)*dh), (0, 0, 255), thickness=2)
        cv.rectangle(img2, (j*dw, i*dh), ((j+1)*dw, (i+1)*dh), (0, 0, 255), thickness=2)


# Display the images
cv.imshow('image1', img1)
cv.imshow('image2', img2)

# Create a blank image with the same dimensions as the input image
mask = np.zeros_like(img2)


# Plot a histogram for each region
# A histogram is a graphical representation of the distribution of pixel intensities in an image.
fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(10, 10))
diff_list=[]
for i in range(3):
    for j in range(3):
        # Set the mask for the current region
        mask.fill(0)
        mask[i*dh:(i+1)*dh, j*dw:(j+1)*dw] = 255
        
        # Calculate the histogram for each image using the current mask
        hist1 = cv.calcHist([img1],[0],mask,[256],[0,256])
        hist2 = cv.calcHist([img2],[0],mask,[256],[0,256])
        diff=hist1-hist2
        diff_list.append(diff)
        # Plot the histograms for the current region
        axs[i, j].plot(hist1, color='red', label='Image 1')
        axs[i, j].plot(hist2, color='blue', label='Image 2')
        axs[i, j].plot(diff, color='green', label='DIFF')
        axs[i, j].set_title(f"Region ({i+1}, {j+1})")
        axs[i, j].legend()
print(len(diff_list))
for seg in diff_list:
    indices = np.where(seg == 0)
    arr = np.delete(seg, indices)
    sum = np.sum(arr)
    print(sum)
    
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.suptitle('Histogram Comparison')
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()