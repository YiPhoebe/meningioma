
from glob import glob
import os

for phase in ["n_test", "n_train", "n_val"]:
    path = f"/Users/iujeong/03_meningioma/3.normalize/{phase}/nii"
    files = glob(os.path.join(path, "*_norm.nii.gz"))
    patient_ids = set(os.path.basename(f).replace("_norm.nii.gz", "") for f in files)
    print(f"{phase}: {len(patient_ids)}명")