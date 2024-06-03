import dicom2nifti
import pandas as pd
import os


import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from pathlib import Path

base_path = "/dtu/p1/johlau/Thesis-Synthex/"
data_path = "data/RAD-ChestCT-NIFTI/"
output_path = "data/RAD-ChestCT-DRR/"


# volumes = os.listdir(base_path + data_path)
# carm = deepdrr.MobileCArm(sensor_height=2250, sensor_width=2250, pixel_size=0.2)

# for idx, volume in enumerate(volumes):
#     patient = deepdrr.Volume.from_nifti(
#         Path(base_path + data_path + volume), 
#         use_thresholding=False,
#     )


#     with Projector(patient, carm=carm) as projector:

#         patient.orient_patient(head_first=True, supine=True)
#         patient.place_center(carm.isocenter_in_world)

#         carm.move_to(isocenter_in_world=patient.center_in_world + geo.v(-25, 0, -500))
#         image = projector()

#     path = Path(base_path + output_path + f"{volume}.png")
#     image_utils.save(path, image)

#     print(f"saved example projection image to {path.absolute()}")



def sample_at_angle(
        input_path,
        alpha, 
        beta
    ):

    carm = deepdrr.MobileCArm(sensor_height=2500, sensor_width=2500, pixel_size=0.2)

    patient = deepdrr.Volume.from_nifti(
        Path(input_path), 
        use_thresholding=True,
    )

    with Projector(patient, carm=carm) as projector:
        carm.move_to(isocenter_in_world=patient.center_in_world + geo.v(0, 0, -750)     
        )

        # carm.move_by(
        #     delta_alpha=alpha,
        #     delta_beta=beta,
        # )
        image = projector()

    return image

if __name__ == "__main__":


    image = sample_at_angle(
        input_path="/dtu/p1/johlau/Thesis-Synthex/data/RAD-ChestCT-NIFTI/trn00067.nii.gz",
        alpha=10,
        beta=10
    )

    image_utils.save(
        "/dtu/p1/johlau/Thesis-Synthex/test_angle_wider.png", 
        image
    )