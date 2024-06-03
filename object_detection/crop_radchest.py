import pandas as pd
import numpy as np
from tqdm import tqdm
import os
os.environ["KERAS_BACKEND"] = "jax" 

import random
import keras
from VisionTransformer import Patches, PatchEncoder

# Function to change the image paths

def crop_augment(
        im, 
        model,
        model_input_size =224,
        original_image_size=(2000,2000), 
        padding=10
    ):

    padding= random.randint(50,100)
    image = im.resize((model_input_size,model_input_size))
    #_im = keras.utils.img_to_array(m)

    preds = model.predict(np.expand_dims(keras.utils.img_to_array(image), axis=0))[0]

    top_left_x, top_left_y = max(int(preds[0] * original_image_size[1]) - padding , 0) , max(int(preds[1] * original_image_size[0]) - padding , 0)
 
    bottom_right_x, bottom_right_y = min(int(preds[2] * original_image_size[1]), original_image_size[1]) + padding , min(int(preds[3] * original_image_size[0]) + padding ,original_image_size[0]) 

    img_crop = im.crop([top_left_x, top_left_y+200, bottom_right_x, bottom_right_y])

    return img_crop


model = keras.saving.load_model("/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/vit_object_detector_50000_samples.keras")

input_dir = "/dtu/p1/johlau/Thesis-Synthex/data/RAD-ChestCT-DRR/"
output_dir =  "/dtu/p1/johlau/Thesis-Synthex/data/RAD-ChestCT-CROPPED/"
image_paths = os.listdir(input_dir)



for idx, path in tqdm(enumerate(image_paths)):
    #print(source_paths[idx])
    image =  keras.utils.load_img(input_dir + path)
    cropped_image = crop_augment(image,model, original_image_size=image.size )
    #print(output_paths[idx])  
    with open(output_dir + path, "+wb") as file:

    #     print(output_paths[idx])  
        #
        keras.utils.save_img(file, cropped_image)
