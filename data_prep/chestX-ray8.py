import pandas as pd
from PIL import Image
import cv2
import numpy as np
import random
from tqdm import tqdm

def rle2mask(mask_rle: str, label=1, shape=(3520,4280)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)# Needed to align to RLE direction


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_both_lungs(row, label=1):

    right = rle2mask(
        mask_rle=row["Right Lung"],
        label=label,
        shape=(int(row["Height"]),int(row["Width"]))
    )

    left = rle2mask(
        mask_rle=row["Left Lung"],
        label=label,
        shape=(int(row["Height"]),int(row["Width"]))
    )

    return right + left


def bounding_box(image, label=1):
    _image = image.copy()
    segmentation = np.where(_image == label)
    padding = random.randint(100,200)

    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = max(int(np.min(segmentation[1]) - padding), 0)
        x_max = min(int(np.max(segmentation[1]) + padding), len(segmentation[1])-1)
        y_min = max(int(np.min(segmentation[0]) - padding), 0)
        y_max = min(int(np.max(segmentation[0]) + padding), len(segmentation[0])-1)


    return x_min, y_min , x_max, y_max

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def crop_and_resize(image, patient_record,target_size):
    mask = decode_both_lungs(patient_record)
    bbox = bounding_box(mask)

    cropped_image = image.crop(bbox)
    cropped_image.thumbnail(target_size)

    cropped_resized_image = expand2square(cropped_image, 0)

    return cropped_resized_image

input_dir = "/home/data_shares/purrlab_students/ChestX-ray14/"
output_dir = "/home/caap/Thesis-Synthex/data/chestx-ray14/"

chestx_ray_df = pd.read_csv("/home/caap/Thesis-Synthex/chestx_ray14_test.csv")

for i in tqdm(range(len(chestx_ray_df))):
    patient = chestx_ray_df .iloc[i]
    image = Image.open(input_dir + patient["Folder"] +"/"+patient["Image Index"])

    cropped_and_resized_image = crop_and_resize(
        image,
        patient,
        (512,512)
    )

    cropped_and_resized_image.save(output_dir + patient["Image Index"])