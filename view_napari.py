import napari
import nibabel as nib
import numpy as np

img = nib.load("embedding_outputs/embedded_tumor_late_volume.nii.gz").get_fdata()
mask = nib.load("embedding_outputs/embedded_tumor_late_mask.nii.gz").get_fdata().astype(np.uint8)

viewer = napari.Viewer()
viewer.add_image(img, name="MRI", colormap="gray")
viewer.add_labels(mask, name="Tumor")

napari.run()
