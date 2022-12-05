import numpy as np
import csv
from PIL import Image
import skimage.transform
import time
import os

'''Converts the compression rle files to images.

Input:
rlefile: csv file containing compressed masks from the segmentation algorithm
outputdirectory: directory to write images to
preprocessed_image_list: csv file containing list of images and their heights and widths'''
def convert_to_image(rlefile, outputdirectory, preprocessed_image_list,
                     rescale = False, scale_factor = 2, verbose = False):

    rle = csv.reader(open(rlefile), delimiter=',')
    rle = np.array([row for row in rle])[1:, :]

    image_list = csv.reader(open(preprocessed_image_list), delimiter=',')
    image_list = np.array([row for row in image_list])[1:, :]

    def rleToMask(rleString,height,width):
      rows,cols = height,width
      rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
      rlePairs = np.array(rleNumbers).reshape(-1,2)
      img = np.zeros(rows*cols,dtype=np.uint8)
      for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
      img = img.reshape(cols,rows)
      img = img.T
      return img

    if outputdirectory[-1] != "/":
        outputdirectory = outputdirectory + "/"
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)

    files = np.unique(rle[:, 0])
    for f in files:
        if verbose:
            start_time = time.time()
            print ("Converting", f, "to mask...")

        list_index = np.where(image_list[:, 0] == f)[0][0]
        file_string = image_list[list_index, 1]

        size = file_string.split(" ")
        height = np.int(size[1])
        width = np.int(size[2])

        new_height = height
        new_width = width
        if rescale:
            new_height = np.int(height // scale_factor)
            new_width = np.int(width // scale_factor)

        image = np.zeros((new_height, new_width)).astype(np.float32)
        columns = np.where(rle[:, 0] == f)
        currobj = 1
        for i in columns[0]:
            currimg = rleToMask(rle[i, 1], new_height, new_width)
            currimg = currimg > 1
            image = image + (currimg * currobj)
            currobj = currobj + 1

        if rescale:
            image = skimage.transform.resize(image, output_shape = (height, width), order = 0, preserve_range = True)

        image = Image.fromarray(image)
        image.save(outputdirectory + f + ".tif")

        if verbose:
            print ("Completed in", time.time() - start_time)
