import numpy as np
from PIL import Image
import os

input1 = str(input('Give Input 1'))
input2 = str(input('give Input 2'))

# Path
path = os.getcwd()
 
# Join various path components
path1=os.path.join(path,'data',input1)
path2=os.path.join(path,'data',input2)
# Load the two images
img1 = Image.open(path1)
img2 = Image.open(path2)

# Trim the images to the same size
width = min(img1.size[0], img2.size[0])
height = min(img1.size[1], img2.size[1])
img1 = img1.crop((0, 0, width, height))
img2 = img2.crop((0, 0, width, height))

# Convert the images to grayscale
img1_gray = img1.convert('L')
img2_gray = img2.convert('L')

# Convert the images to ndarrays
img1_arr = np.array(img1_gray)
img2_arr = np.array(img2_gray)

# Subtract the two images
diff_arr = img1_arr - img2_arr

# Convert the ndarray back to an image
diff_img = Image.fromarray(diff_arr)

# Save the image as a PNG file
name=input1[-5]+'_'+input2[-5]
diff_img.save(f'data/sub{name}.png')