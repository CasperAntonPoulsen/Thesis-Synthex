import os
import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from pathlib import Path
import itertools

base_path = "/dtu/p1/johlau/Thesis-Synthex/"
data_path = "data/RAD-ChestCT-NIFTI/"
output_path = "data/RAD-ChestCT-DRR-angled/"

angles = [j for j in itertools.product([i for i in range(-10, 15, 5)], [i for i in range(-10, 15, 5)]) if not ((5 in j and 0 in j) or (-5 in j and 0 in j) or (j == (0,0)))]

volumes = os.listdir(base_path+data_path)
print(volumes)
for idx, volume in enumerate(volumes):
    if not volume.endswith(".nii.gz"):
        continue

    patient = deepdrr.Volume.from_nifti(
        Path(base_path + data_path + volume), 
        use_thresholding=False,
    )

    for angle in angles:

        carm = deepdrr.MobileCArm(
            sensor_height=2500, 
            sensor_width=2500,
            source_to_detector_distance=2000,
            source_to_isocenter_vertical_distance=2000,
            pixel_size=0.18
        )


        with Projector(patient, carm=carm) as projector:
            patient.orient_patient(head_first=True, supine=True)
            patient.place_center(carm.isocenter_in_world)

            carm.move_by(
                delta_alpha=angle[0],
                delta_beta=angle[1],
            )

            image = projector()

        path = Path(base_path + output_path + f"{volume}_{angle[0]}_{angle[1]}.png")
        image_utils.save(path, image)

        print(f"saved example projection image to {path.absolute()}")
