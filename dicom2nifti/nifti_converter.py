from dicom2nifti import convert_directory
import pandas as pd
import os

base_path = "/dtu/p1/johlau/Thesis-Synthex/"
data_path = "data/LIDC-IDRI-raw/"
output_path = "data/LIDC-IDRI-NIFTI/"

files = os.listdir(base_path + data_path)

df_metadata = pd.read_csv(base_path + data_path + [file for file in files if file.endswith(".csv")][-1])

df_ct = df_metadata[df_metadata["Modality"] == "CT"].reset_index(drop=True)

for i in range(0, len(df_ct)):

    convert_directory(base_path+data_path+df_ct.iloc[i]["SeriesInstanceUID"], base_path+output_path,base_filename=df_ct.iloc[i]["PatientID"], reorient=False)


    file_names = [k for k in os.listdir(base_path+output_path) if not k.startswith("LIDC-IDRI")]
    #os.rename(base_path+output_path+file_names[0], base_path+output_path+df_ct.iloc[i]["PatientID"] + ".nii.gz")