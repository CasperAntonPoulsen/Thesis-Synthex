import nibabel as nib
import deepdrr
import reorient_nii


test = nib.load("/dtu/p1/johlau/Thesis-Synthex/data/LIDC-IDRI-NIFTI/LIDC-IDRI-0001.nii.gz")
print(test.affine)

print(deepdrr.geo.FrameTransform(test.affine))

print(nib.aff2axcodes(test.affine))

test = reorient_nii.load("/dtu/p1/johlau/Thesis-Synthex/data/1.nii.gz")

print(test.affine)

print(deepdrr.geo.FrameTransform(test.affine))

print(nib.aff2axcodes(test.affine))

test  = reorient_nii.reorient(test, 'LPI')

print(test.affine)

print(deepdrr.geo.FrameTransform(test.affine))

print(nib.aff2axcodes(test.affine))

nib.save(test, "/dtu/p1/johlau/Thesis-Synthex/data/LIDC-IDRI-NIFTI/LIDC-IDRI-0001-LPI.nii.gz")