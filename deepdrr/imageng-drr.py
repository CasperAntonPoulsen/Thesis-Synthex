import pandas as pd
import os


import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from pathlib import Path

base_path = "/dtu/p1/johlau/Thesis-Synthex/"
data_path = "data/ImagEng-NIFTI/"
output_path = "data/ImagEng-DRR/"


volumes = os.listdir(base_path + data_path)
carm = deepdrr.MobileCArm(sensor_height=2000, sensor_width=2000, pixel_size=0.2)

for idx, volume in enumerate(volumes):
    patient = deepdrr.Volume.from_nifti(
        Path(base_path + data_path + volume), 
        use_thresholding=False,
    )

    with Projector(patient, carm=carm) as projector:

        patient.orient_patient(head_first=False, supine=False)
        patient.place_center(carm.isocenter_in_world)

        carm.move_to(isocenter_in_world=patient.center_in_world + geo.v(-25, 0, -500))
        image = projector()

    path = Path(base_path + output_path + f"{idx+1}.png")
    image_utils.save(path, image)

    print(f"saved example projection image to {path.absolute()}")
    
