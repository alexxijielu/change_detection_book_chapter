import os
import csv
import numpy as np
import subprocess
import shutil

from .mrcnn.my_inference import predict_images
from .mrcnn.preprocess_images import preprocess_images_with_string, preprocess_images
from .mrcnn.convert_to_image import convert_to_image

from PIL import Image
import skimage.transform
import skimage.exposure
import scipy.ndimage.measurements as ndi
import mahotas
import cv2

def segment_directory(input_dir, output_dir, verbose=False, rescale=True, scale_factor=2.0, contains_str=None):
    if output_dir != '' and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    preprocessed_image_directory = output_dir + "/preprocessed_images/"
    preprocessed_image_list = output_dir + "/preprocessed_images_list.csv"
    rle_file = output_dir + "/compressed_masks.csv"
    output_mask_directory = output_dir + "/masks/"

    # Preprocess the images
    if verbose:
        print("\nPreprocessing images...")
    if contains_str:
        preprocess_images_with_string(input_dir,
                          preprocessed_image_directory,
                          preprocessed_image_list,
                          verbose=verbose,
                          string=contains_str)
    else:
        preprocess_images(input_dir,
                          preprocessed_image_directory,
                          preprocessed_image_list,
                          verbose=verbose)

    if verbose:
        print("\nRunning your images through the neural network...")
    predict_images(preprocessed_image_directory,
                   preprocessed_image_list,
                   rle_file,
                   rescale=rescale,
                   scale_factor=scale_factor,
                   verbose=verbose)

    if verbose:
        print("\nSaving the masks...")
    convert_to_image(rle_file,
                     output_mask_directory,
                     preprocessed_image_list,
                     rescale=rescale,
                     scale_factor=scale_factor,
                     verbose=verbose)

    os.remove(preprocessed_image_list)
    shutil.rmtree(preprocessed_image_directory)
    os.remove(rle_file)
    print ("Completed segmentation!")


def crop_single_cells(filename, image_dir, mask_dir, out_dir, additional_img_strs, cropsize=64, resize=64):
    mask = np.array(Image.open(mask_dir + filename))
    mask = mahotas.labeled.remove_bordering(mask)
    max_cells = np.int(np.max(mask))

    for cell in range(1, max_cells):
        temp_img = np.zeros((mask.shape[0], mask.shape[1]))
        temp_img[np.where(mask == cell)] = 1
        if np.sum(temp_img) >= 10:
            y, x = ndi.center_of_mass(temp_img)

            # Calculate the crop
            x = np.int(np.round(x))
            y = np.int(np.round(y))
            c1 = y - cropsize // 2
            c2 = y + cropsize // 2
            c3 = x - cropsize // 2
            c4 = x + cropsize // 2

            main_img_name = image_dir + filename
            main_img = np.array(Image.open(main_img_name))

            # try to get the crop and save it as rgb tif; skip cells that can't get a full crop for

            if c1 < 0 or c3 < 0 or c2 > main_img.shape[0] or c4 > main_img.shape[1]:
                pass
            else:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                # Crop main image and save as single cell crop
                out_img = main_img[c1:c2, c3:c4]
                out_img = skimage.transform.resize(out_img, (resize, resize), preserve_range=True)
                basename = os.path.splitext(filename)[0]

                ext = os.path.splitext(filename)[1]
                out_img = Image.fromarray(out_img)
                out_img.save(out_dir + basename + "_" + str(cell) + ext)

                # Crop any additional image using provided string replacements
                for str_rep in additional_img_strs:
                    addi_name = filename.replace(str_rep[0], str_rep[1])
                    addi_basename = os.path.splitext(addi_name)[0]
                    ext = os.path.splitext(addi_name)[1]
                    addi_img = np.array(Image.open(image_dir + filename.replace(str_rep[0], str_rep[1])))
                    addi_crop = addi_img[c1:c2, c3:c4]
                    addi_crop = skimage.transform.resize(addi_crop, (resize, resize), preserve_range=True)
                    addi_crop = Image.fromarray(addi_crop)
                    addi_crop.save(out_dir + addi_basename + "_" + str(cell) + ext)


def batch_crop_single_cells(image_dir, mask_dir, output_dir, additional_img_strs, cropsize=64, resize=64):

    for filename in os.listdir(mask_dir):
        print ("Working on", filename)
        crop_single_cells(filename,
                          image_dir,
                          mask_dir,
                          output_dir,
                          additional_img_strs,
                          cropsize,
                          resize)
    print ("Successfully cropped all images.")
