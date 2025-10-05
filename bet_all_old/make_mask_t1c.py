#  t1c_bet.nii.gz 파일에 대해 binary 마스크(_bet_mask.nii.gz)를 만드는 코드
# 저장위치 :
# /Users/iujeong/03_meningioma/1.bet_all_old/b_test
# /Users/iujeong/03_meningioma/1.bet_all_old/b_train
# /Users/iujeong/03_meningioma/1.bet_all_old/b_val


import pandas as pd
import nibabel as nib
import numpy as np
import os

patient_ids_df = pd.read_csv("/Users/iujeong/03_meningioma/bet_all_old/patient_ids_by_split.csv")
patient_ids_df['split'] = patient_ids_df['split'].str.replace('r_', '')
test_ids = set(patient_ids_df[patient_ids_df['split'] == 'test']["patient_id"])
train_ids = set(patient_ids_df[patient_ids_df['split'] == 'train']["patient_id"])
val_ids = set(patient_ids_df[patient_ids_df['split'] == 'val']["patient_id"])

def extract_pid(filename):
    return filename.split("_")[0]

bet_dirs = [
    "/Users/iujeong/03_meningioma/bet_all_old/b_train",
    "/Users/iujeong/03_meningioma/bet_all_old/b_val",
    "/Users/iujeong/03_meningioma/bet_all_old/b_test",
]

for bet_dir in bet_dirs:
    all_files = [f for f in os.listdir(bet_dir) if f.endswith("_bet.nii.gz") and not f.startswith(".")]

    for f in all_files:
        pid = extract_pid(f)
        if pid in train_ids:
            subfolder = "b_train"
        elif pid in val_ids:
            subfolder = "b_val"
        else:
            subfolder = "b_test"

        filepath = os.path.join(bet_dir, f)
        try:
            img = nib.load(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        out_dir = os.path.join("/Users/iujeong/03_meningioma/bet_all_old", subfolder)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f.replace("_bet.nii.gz", "_bet_mask.nii.gz"))
        if os.path.exists(out_path):
            continue

        data = img.get_fdata()
        mask = (data > 0).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask, img.affine)
        nib.save(mask_img, out_path)