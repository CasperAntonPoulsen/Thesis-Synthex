import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

input_dir = "/home/data_shares/purrlab/padchest/"
output_dir =  "/home/caap/Thesis-Synthex/data/padchest-resized/"
files =  os.listdir(input_dir )
df = pd.read_csv(input_dir+[i for i in files if i.endswith(".csv")][0])
df = df[df["Projection"] == "PA"]

print(len(df))

for idx in tqdm(range(len(df))):
    try:
        record = df.iloc[idx]
        input_path = input_dir+str(record["ImageDir"])+ "/"+ record["ImageID"]
        output_path = output_dir+str(record["ImageDir"])+ "-"+ record["ImageID"]

        image = Image.open(input_path)

        image.thumbnail((1024,1024))
        #print(output_paths[idx])  
        with open(output_path, "+wb") as file:

        #     print(output_paths[idx])  
            #
            image.save(file)
    except Exception:
        continue
