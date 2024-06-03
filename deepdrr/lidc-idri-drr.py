import dicom2nifti
import pandas as pd
import os


import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from pathlib import Path



base_path = "/mnt/c/Users/caspe/Thesis-Synthex/"
data_path = "data/LIDC-IDRI-raw/"


files = os.listdir(base_path + data_path)
df_metadata = pd.read_csv(base_path + data_path + [file for file in files if file.endswith(".csv")][-1])

patients = list(df_metadata[df_metadata["Modality"] == "CT"]["PatientID"])


base_path = "/mnt/c/Users/caspe/Thesis-Synthex/"
data_path = "data/LIDC-IDRI-NIFTI/"
output_path = "data/LIDC-IDRI-DRR/"


volumes = os.listdir(base_path + data_path)
carm = deepdrr.MobileCArm(sensor_height=2250, sensor_width=2250, pixel_size=0.2)

for idx, volume in enumerate(patients):
    if volume + ".nii.gz" not in volumes:
        continue

    patient = deepdrr.Volume.from_nifti(
        Path(base_path + data_path + volume + ".nii.gz"), 
        use_thresholding=False,
    )


    with Projector(patient, carm=carm) as projector:

        patient.orient_patient(head_first=False, supine=False)
        patient.place_center(carm.isocenter_in_world)

        carm.move_to(isocenter_in_world=patient.center_in_world + geo.v(-25, 0, -475))
        image = projector()

    path = Path(base_path + output_path + f"{volume}.png")
    image_utils.save(path, image)

    print(f"saved example projection image to {path.absolute()}")
