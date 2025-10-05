# ========================================================
# ✅ 모델 학습용 슬라이스 저장 (Z-score 정규화 포함, 크기 보정 X)
# - HD-BET으로 스컬 제거된 T1c 이미지 기준
# - intensity normalization (Z-score, brain 영역 기준) 적용
# - 크기 보정은 적용하지 않음 (모델 입력 시 처리 예정)
# ========================================================
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import datetime


re_dirs = [
    "/Users/iujeong/03_meningioma/2.resample/r_test",
    "/Users/iujeong/03_meningioma/2.resample/r_train",
    "/Users/iujeong/03_meningioma/2.resample/r_val",
]

all_gtv_stats = []
all_bbox_stats = []

all_cases = []
for d in re_dirs:
    phase = d.split("/")[-1].replace("r_", "")  # extract phase from path
    for f in sorted(glob(os.path.join(d, "*_t1c_gtv_mask.nii.gz"))):
        pid = os.path.basename(f).replace("_t1c_gtv_mask.nii.gz", "")
        # Clean up pid if it ends with "_t1c"
        if pid.endswith("_t1c"):
            pid = pid.replace("_t1c", "")
        all_cases.append((pid, d, phase))

for pid, re_dir, phase in all_cases:
    gtv_clipping_stats = []
    bbox_stats = []

    # 각 슬라이스의 크기 기록 리스트
    slice_shape_stats = []

    group_dir = os.path.join("/Users/iujeong/03_meningioma/3.normalize", f"n_{phase}", "nii")

    img_path = os.path.join(re_dir, f"{pid}_t1c_bet.nii.gz")
    gtv_path = os.path.join(re_dir, f"{pid}_t1c_gtv_mask.nii.gz")
    bet_path = os.path.join(re_dir, f"{pid}_t1c_bet_mask.nii.gz")

    if not os.path.exists(img_path) or not os.path.exists(gtv_path) or not os.path.exists(bet_path):
        print(f"{pid}: 필요한 파일 없음")
        print(f"  - img_path: {img_path}")
        print(f"  - gtv_path: {gtv_path}")
        print(f"  - bet_path: {bet_path}")
        continue

    img = nib.load(img_path).get_fdata()
    bet_mask = nib.load(bet_path).get_fdata()

    # 이미지 크기도 맞춰줘야 정규화 가능
    if img.shape != bet_mask.shape:
        adjusted_img = np.zeros_like(bet_mask, dtype=img.dtype)
        x = min(img.shape[0], bet_mask.shape[0])
        y = min(img.shape[1], bet_mask.shape[1])
        z = min(img.shape[2], bet_mask.shape[2])
        adjusted_img[:x, :y, :z] = img[:x, :y, :z]
        img = adjusted_img

    # BET 마스크가 비었는지 검사
    if np.sum(bet_mask) == 0:
        print(f"{pid}: BET 마스크가 비어 있음 (skip)")
        # 로그 저장
        os.makedirs("/Users/iujeong/03_meningioma/8.result/log", exist_ok=True)
        with open("/Users/iujeong/03_meningioma/8.result/log/skipped_cases.txt", "a") as log_file:
            log_file.write(f"{pid}: BET 마스크가 비어 있음\n")
        continue

    # [추가] 전체 brain bounding box 계산
    x_any, y_any = np.any(bet_mask, axis=(1, 2)), np.any(bet_mask, axis=(0, 2))
    x_min, x_max = np.where(x_any)[0][[0, -1]]
    y_min, y_max = np.where(y_any)[0][[0, -1]]

    # margin 추가
    margin = 5
    x_min = max(0, x_min - margin)
    x_max = min(bet_mask.shape[0], x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(bet_mask.shape[1], y_max + margin)

    # bounding box 정보 저장
    bbox_stats.append({
        "patient_id": pid,
        "phase": phase,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max
    })

    gtv_mask = nib.load(gtv_path).get_fdata()

    # shape mismatch 처리: BET 마스크를 이미지 크기에 맞게 pad 또는 crop
    if img.shape != bet_mask.shape:
        target_shape = img.shape
        current_shape = bet_mask.shape
        adjusted = np.zeros(target_shape, dtype=bet_mask.dtype)

        # 공통 부분은 복사
        x = min(target_shape[0], current_shape[0])
        y = min(target_shape[1], current_shape[1])
        z = min(target_shape[2], current_shape[2])
        adjusted[:x, :y, :z] = bet_mask[:x, :y, :z]

        bet_mask = adjusted

    # shape matching: gtv_mask and bet_mask 모두 img 기준으로 맞춤
    def match_shape(mask, target_shape):
        adjusted = np.zeros(target_shape, dtype=mask.dtype)
        x = min(mask.shape[0], target_shape[0])
        y = min(mask.shape[1], target_shape[1])
        z = min(mask.shape[2], target_shape[2])
        adjusted[:x, :y, :z] = mask[:x, :y, :z]
        return adjusted

    if gtv_mask.shape != img.shape:
        print(f"{pid}: gtv_mask shape mismatch → {gtv_mask.shape} → {img.shape}")
        gtv_mask = match_shape(gtv_mask, img.shape)

    if bet_mask.shape != img.shape:
        print(f"{pid}: bet_mask shape mismatch → {bet_mask.shape} → {img.shape}")
        bet_mask = match_shape(bet_mask, img.shape)

    print(f"{pid} shapes → img: {img.shape}, bet: {bet_mask.shape}, gtv: {gtv_mask.shape}")
    os.makedirs("/Users/iujeong/03_meningioma/8.result/log", exist_ok=True)
    with open("/Users/iujeong/03_meningioma/8.result/log/shape_check_log.txt", "a") as f:
        f.write(f"{pid} shapes → img: {img.shape}, bet: {bet_mask.shape}, gtv: {gtv_mask.shape}\n")

    # GTV 바깥 제거
    original_gtv_voxels = np.sum(gtv_mask > 0)
    gtv_mask = gtv_mask * (bet_mask > 0)
    kept_gtv_voxels = np.sum(gtv_mask > 0)
    removed_ratio = 1 - (kept_gtv_voxels / original_gtv_voxels) if original_gtv_voxels > 0 else 0
    gtv_clipping_stats.append({
        "patient_id": pid,
        "phase": phase,
        "original": int(original_gtv_voxels),
        "kept": int(kept_gtv_voxels),
        "removed_ratio": removed_ratio
    })

    # === [1] Intensity Normalization (Z-score, brain 중간값 기반) ===
    # brain_pixels = img[bet_mask > 0]
    # if brain_pixels.size < 10:
    #     print(f"{pid}: Brain pixel too small for normalization (skip)")
    #     continue
    # low, high = np.percentile(brain_pixels, [10, 90])
    # trimmed = brain_pixels[(brain_pixels >= low) & (brain_pixels <= high)]
    # mean, std = trimmed.mean(), trimmed.std()
    # img = (img - mean) / (std + 1e-8) 
    brain_pixels = img[bet_mask > 0]
    if brain_pixels.size < 10:
        print(f"{pid}: Brain pixel too small for normalization (skip)")
        continue
    mean, std = brain_pixels.mean(), brain_pixels.std()
    img = (img - mean) / (std + 1e-8)

    # === 마지막 shape 맞추기: 모든 마스크를 img 기준으로 pad 또는 crop ===


    # 정규화된 볼륨 저장
    normalized_save_dir = f"/Users/iujeong/03_meningioma/3.normalize/n_{phase}"
    png_dir = os.path.join(normalized_save_dir, "png")
    nii_dir = os.path.join(normalized_save_dir, "nii")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(nii_dir, exist_ok=True)

    import shutil

    # 마스크도 같은 위치로 복사 (정규화는 안 함)
    orig_bet_mask_path = os.path.join(re_dir, f"{pid}_t1c_bet_mask.nii.gz")
    orig_gtv_mask_path = os.path.join(re_dir, f"{pid}_t1c_gtv_mask.nii.gz")
    # BET 마스크: 타입을 np.uint8로 변환해서 저장
    orig_bet_mask = nib.load(orig_bet_mask_path)
    orig_bet_data = orig_bet_mask.get_fdata().astype(np.uint8)
    bet_img = nib.Nifti1Image(orig_bet_data, orig_bet_mask.affine)
    nib.save(bet_img, os.path.join(nii_dir, f"{pid}_bet_mask.nii.gz"))
    # GTV 마스크는 그대로 복사
    shutil.copy(orig_gtv_mask_path, os.path.join(nii_dir, f"{pid}_gtv_mask.nii.gz"))

    nii = nib.load(img_path)  # 원래 이미지에서 affine 가져옴
    norm_nii_path = os.path.join(nii_dir, f"{pid}_norm.nii.gz")
    n_img = nib.Nifti1Image(img, affine=nii.affine)
    n_img = nib.as_closest_canonical(n_img)
    nib.save(n_img, norm_nii_path)

    # PNG 저장용으로 NIfTI 다시 불러와서 시각화
    img_from_nii = nib.load(norm_nii_path).get_fdata()
    for i in range(img_from_nii.shape[2]):
        slice_img = img_from_nii[:, :, i]
        brain_pixels_slice = slice_img[bet_mask[:, :, i] > 0]
        if brain_pixels_slice.size == 0:
            continue  # skip 빈 마스크
        vmin, vmax = np.percentile(brain_pixels_slice, [1, 99])
        filename = os.path.join(png_dir, f"{pid}_slice_{i:03d}.png")
        plt.imsave(filename, slice_img, cmap="gray", vmin=vmin, vmax=vmax)
        plt.close()

    # 슬라이스 분할은 다른 단계에서 수행

    plt.hist(brain_pixels, bins=100, alpha=0.5, label='Raw')
    plt.legend()
    plt.title(f"{pid} - Brain Intensity Distribution")
    hist_dir = "/Users/iujeong/03_meningioma/8.result/hist"
    os.makedirs(hist_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = os.path.join(hist_dir, f"{pid}_brain_hist_{timestamp}.png")
    plt.savefig(hist_path)
    plt.close()

    # plt.show()

    print(f"{pid}: 저장 완료")

    all_gtv_stats.extend(gtv_clipping_stats)
    all_bbox_stats.extend(bbox_stats)

# GTV 클리핑 통계 저장
os.makedirs("/Users/iujeong/03_meningioma/8.result/log", exist_ok=True)
df = pd.DataFrame(all_gtv_stats)
df.to_csv("/Users/iujeong/03_meningioma/8.result/csv/gtv_clipping_stats.csv", index=False)

# Bounding Box 저장
df_bbox = pd.DataFrame(all_bbox_stats)
df_bbox.to_csv("/Users/iujeong/03_meningioma/8.result/csv/bbox_stats.csv", index=False)
print("📌 Using bbox CSV:", df_bbox)
