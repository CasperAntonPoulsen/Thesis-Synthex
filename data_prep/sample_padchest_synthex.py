import pandas as pd
from PIL import Image
import numpy as np
import os
import ast
from tqdm import tqdm

input_dir = "/home/caap/Thesis-Synthex/data/padchest-resized/"
image_files = os.listdir(input_dir)


metadata_dir = "/home/data_shares/purrlab/padchest/"
files= os.listdir(metadata_dir)

label_dict = {
    'pleural effusion': 'Effusion', 
    'pneumothorax': 'Pneumothorax', 
    'atelectasis': 'Atelectasis', 
    'cardiomegaly':'Cardiomegaly', 
    'pneumonia':'Pneumonia',
}

padchest_df = pd.read_csv(metadata_dir +[i for i in files if i.endswith(".csv")][0])
padchest_df = padchest_df[padchest_df["Projection"] == "PA"]
padchest_df = padchest_df[~padchest_df["Labels"].isna()]
padchest_df["Label_list"] = padchest_df["Labels"].apply(lambda x : ast.literal_eval(x))

def label_onehot_encoder(row, label, column_name):
    labels = row[column_name]
    if label in labels:
        return 1
    else:
        return 0

for label in label_dict.keys():
    padchest_df[label_dict[label]] = padchest_df.apply(label_onehot_encoder, args=(label, "Label_list"), axis=1)

padchest_df["FileID"] = padchest_df["ImageDir"].apply(lambda x : str(x)) + "-" + padchest_df["ImageID"]
padchest_df = padchest_df[['FileID', 'Effusion', 'Pneumothorax', 'Atelectasis',
       'Cardiomegaly', 'Pneumonia']]

padchest_df=padchest_df.loc[padchest_df[padchest_df[['Effusion', 'Pneumothorax', 'Atelectasis','Cardiomegaly', 'Pneumonia']] > 0 ].dropna(how="all").index]

out_dir = "/home/caap/Thesis-Synthex/trainB/"

for file in tqdm(padchest_df["FileID"].reset_index(drop=True)):
    image = Image.open(input_dir+file)
    greyscale_image = Image.fromarray(np.array(image)/255).convert("L")
    greyscale_image.save(out_dir+file)