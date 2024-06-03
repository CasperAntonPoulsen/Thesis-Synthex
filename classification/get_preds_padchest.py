# Imports
import pandas as pd
import numpy as np
#import tensorflow as tf
import os
import sys

os.environ["KERAS_BACKEND"] = "jax" 
import keras
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from tqdm import tqdm

# Function for creating the n-hot encoding
def get_n_hot_encoding(df, labels_to_encode):
    enc = np.zeros((len(df), len(labels_to_encode)))
    for idx, row in df.iterrows():
        for ldx, l in enumerate(labels_to_encode):
            if row[l] == 1:
                enc[idx][ldx] = 1
    return enc


def main():
    model_dir = "/dtu/p1/johlau/Thesis-Synthex/models/"
    models = os.listdir(model_dir)
    models = [i for i in models if i.endswith(".keras") and (i.startswith("drr") or i.startswith("baseline"))]

    data_dir = "/dtu/p1/johlau/Thesis-Synthex/data/"
    output_dir = "/dtu/p1/johlau/Thesis-Synthex/predictions/"


    padchest_test = pd.read_csv(data_dir+ "padchest_dev.csv")


    x_test = np.array([keras.utils.img_to_array(keras.utils.load_img(i, color_mode="grayscale").convert("RGB")) for i in tqdm(padchest_test["image_path"])])/255
    
    labels_to_encode = ['Effusion', 'Pneumothorax', 'Atelectasis', 'Cardiomegaly', 'Pneumonia']
    padchest_labels = get_n_hot_encoding(padchest_test, labels_to_encode)

    # MT
    for name in models:
        model = keras.saving.load_model(model_dir + name)


        predictions = model.predict(x_test)

        pred_df = pd.DataFrame({
            labels_to_encode[i] : predictions[:, i] for i in range(5)
        })

        pred_df.to_json(output_dir + "padchest_" + name.replace(".keras", ".json"))



if __name__ == "__main__":
    main()