import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

def main():
    # vis_dir will now be determined per input_dir below

    input_dirs = [
        "/Users/iujeong/03_meningioma/4.slice/s_test/npy",
        "/Users/iujeong/03_meningioma/4.slice/s_train/npy",
        "/Users/iujeong/03_meningioma/4.slice/s_val/npy",
    ]

    for input_dir in input_dirs:
        split_name = os.path.basename(os.path.dirname(input_dir))
        vis_dir = f"/Users/iujeong/03_meningioma/4.slice/{split_name}/png"
        os.makedirs(vis_dir, exist_ok=True)

        img_paths = sorted(glob(os.path.join(input_dir, "*_img.npy")))
        for img_path in tqdm(img_paths, desc=f"Visualizing {split_name}"):
            img = np.load(img_path)
            print(f"👉 {os.path.basename(img_path)}: min={img.min():.3f}, max={img.max():.3f}, shape={img.shape}")
            base = os.path.basename(img_path).replace("_img.npy", "")
            save_path = os.path.join(vis_dir, f"{base}.png")
            plt.imsave(save_path, img, cmap='gray')

if __name__ == "__main__":
    main()