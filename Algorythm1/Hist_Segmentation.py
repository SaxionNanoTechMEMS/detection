import os
import pathlib
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sc
from tkinter import Tk
from tkinter.filedialog import askopenfilename

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

        
def calc_metrics(array):

    THRESH_HOLD_STD=6.5
    THRESH_HOLD_IQR=1
    THRESH_HOLD_MAX=40


    numpy = np.array(array)
    outputDict = {}
    """
    The first loop calculates the standard deviation of each image using np.std(numpy, axis=1) 
    and checks if the standard deviation is greater than THRESH_HOLD_STD. 
    If the standard deviation is greater than THRESH_HOLD_STD, it sets the corresponding value in outputDict to 1, 
    indicating that the image has a potential defect. Otherwise, it sets the value to 0."""

    for count,dev in enumerate(np.std(numpy,axis = 1)):
        if(dev > THRESH_HOLD_STD):
            outputDict.setdefault(count,[]).append(1)
            #print(dev, f'Pic nr: {count} has potential defect -----------' ) 
        else:
            outputDict.setdefault(count,[]).append(0)
            #print(dev, f'Pic nr: {count} no defects detected' ) 

    """
    The second loop calculates the interquartile range (IQR) of each image using sc.iqr(numpy, axis=1) 
    and checks if the IQR is greater than or equal to THRESH_HOLD_IQR.
    If the IQR is greater than or equal to THRESH_HOLD_IQR, it sets the corresponding value in outputDict to 1,
    indicating that the image has a potential defect. Otherwise, it sets the value to 0.
    """

    for count,iqr in enumerate(sc.iqr(numpy,axis = 1)):
        if(iqr >= THRESH_HOLD_IQR):
            #print(iqr, f'Pic nr: {count} has potential defect -----------' ) 
            outputDict.setdefault(count,[]).append(1)
        else:
            #print(iqr, f'Pic nr: {count} no defects detected' ) 
            outputDict.setdefault(count,[]).append(0)

    """
    The third loop calculates the maximum absolute value of each image using np.amax(np.absolute(numpy), axis=1)
    and checks if the maximum absolute value is greater than or equal to THRESH_HOLD_MAX.
    If the maximum absolute value is greater than or equal to THRESH_HOLD_MAX, it sets the corresponding value in outputDict to 1, 
    indicating that the image has a potential defect. Otherwise, it sets the value to 0.
    """

    for count,maxVal in enumerate(np.amax(np.absolute(numpy),axis = 1)):
        if(maxVal >= THRESH_HOLD_MAX):
            #print(maxVal, f'Pic nr: {count} has potential defect -----------') 
            outputDict.setdefault(count,[]).append(1)
        else:
            #print(maxVal, f'Pic nr: {count} no defects detected' ) 
            outputDict.setdefault(count,[]).append(0)

    """Export the potential defects ."""
            
    for k in outputDict.keys():
        if np.sum(np.array(outputDict[k])) >= 2:
            print(f'{k} potential problem')

    
def process_image(model_reference, image_path: pathlib.Path):
    # root = Tk()
    # root.withdraw()
    # filename = askopenfilename()

    # Read images and convert them to grayscale
    image = cv.imread(str(image_path.resolve()), 0)

    # Reseize the images so i can do operations later on
    SIZE = 500

    model = cv.resize(model_reference, (SIZE, SIZE))
    image = cv.resize(image, (SIZE, SIZE))

    # Draw a 3X3 rechtangle on the images
    h, w = model.shape[:2]
    splitCount = 10

    dh = h // splitCount
    dw = w // splitCount
    for i in range(splitCount):
        for j in range(splitCount):
            # print(i,j)
            image1 = cv.rectangle(model, (j*dw, i*dh), ((j+1)*dw, (i+1)*dh), (0, 0, 255), thickness=2)
            image2 = cv.rectangle(image, (j*dw, i*dh), ((j+1)*dw, (i+1)*dh), (0, 0, 255), thickness=2)
            
            cv.putText(image1, str(splitCount*i+j), (j*dw + 10, i*dh + int(SIZE/splitCount) ), cv.FONT_HERSHEY_SIMPLEX, 0.9,  (0,0,255), 2)
            cv.putText(image2, str(splitCount*i+j), (j*dw + 10, i*dh + int(SIZE/splitCount) ),cv.FONT_HERSHEY_SIMPLEX, 0.9,   (0,0,255),2)


    # Display the images
    cv.imshow('model', model)
    cv.imshow(image_path.name, image)

    # Create a blank image with the same dimensions as the input image
    mask = np.zeros_like(image)

    # Plot a histogram for each region
    # A histogram is a graphical representation of the distribution of pixel intensities in an image.
    fig, axs = plt.subplots(nrows=splitCount, ncols=splitCount, sharex=True, sharey=True, figsize=(1, 1))
    diff_list=[]
    for i in range(splitCount):
        for j in range(splitCount):
            # Set the mask for the current region
            mask.fill(0)
            mask[i*dh:(i+1)*dh, j*dw:(j+1)*dw] = 255
            
            # Calculate the histogram for each image using the current mask
            hist1 = cv.calcHist([model],[0],mask,[256],[0,256])
            hist2 = cv.calcHist([image],[0],mask,[256],[0,256])
            diff=hist1-hist2
            diff_list.append(diff)
            # Plot the histograms for the current region
            # axs[i, j].plot(hist1, color='red', label='Image 1')
            # axs[i, j].plot(hist2, color='blue', label='Image 2')
            # axs[i, j].plot(diff, color='green', label='DIFF')
            # axs[i, j].set_title(f"Region ({i+1}, {j+1})")
            # axs[i, j].legend()
            #print(len(diff_list))

    #output_diff(diff_list)

    calc_metrics(diff_list)

    for seg in diff_list:
        indices = np.where(seg == 0)
        arr = np.delete(seg, indices)
        sum = np.sum(arr)
        #print(sum)



    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.suptitle('Histogram Comparison')
    # plt.show()

def main():
    model = pathlib.Path().glob('model.png')
    if not model:
        sys.exit("Model image (model.png) is missing")

    test_images = pathlib.Path('test_data').glob('*.png')
    if not test_images:
        sys.exit("Test images (test_data/) is missing")

    test_images = [x.resolve() for x in test_images]


    # Read the model image
    model = str(list(model)[0].resolve())
    model_image = cv.imread(model, 0)

    # Process each test image
    for image in test_images:        
        process_image(model_image, image)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()