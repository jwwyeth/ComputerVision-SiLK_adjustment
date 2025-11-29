import torch
import cv2 as cv
import silk
import os
import numpy as np
import time

# --- Setup device ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load SiLK model ---
model = silk.SiLK()
model = model.to(device)
model.train(False)
model.load_state_dict(torch.load("./train0_25000.pth"))

# --- Keypoint extraction ---
def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=100):
    """Extract top-k keypoints and their descriptors from a SiLK forward pass."""
    height, width = ktps.shape[2:4]
    kpts_map = torch.sigmoid(ktps).squeeze(0).squeeze(0)  # [H, W]
    kpts_map = kpts_map.reshape(height * width)
    desc_map = desc.squeeze(0).reshape(desc.shape[1], height * width)
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice

# --- Multi-scale image pyramid ---
def create_image_pyramid(img, scales=[1.0, 0.75, 0.5]):
    """Return list of grayscale images at different scales."""
    pyramid = []
    for scale in scales:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv.resize(img, (new_w, new_h))
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        pyramid.append((gray, scale))
    return pyramid

# --- Match two images ---
def match_two_multiscale(img0, img1, scales=[1.0, 0.75, 0.5], top_k=200):
    """Compute matches between two images using multi-scale SiLK."""
    pyramid0 = create_image_pyramid(img0, scales)
    pyramid1 = create_image_pyramid(img1, scales)

    all_hw_pairs = []

    for gray0, scale0 in pyramid0:
        img0_tensor = silk.utils.img_to_tensor(gray0, device=device, normalization=True).unsqueeze(0)
        for gray1, scale1 in pyramid1:
            img1_tensor = silk.utils.img_to_tensor(gray1, device=device, normalization=True).unsqueeze(0)

            topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0_tensor), k=top_k)
            topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1_tensor), k=top_k)

            # Normalize descriptors
            topk_desc0 = topk_desc0.permute(1, 0)
            topk_desc1 = topk_desc1.permute(1, 0)
            topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
            topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)

            # Similarity and matching
            sim_mat = torch.matmul(topk_desc0, topk_desc1.T)
            sim_max, sim_indice = torch.max(sim_mat, dim=1)

            height0, width0 = gray0.shape
            height1, width1 = gray1.shape

            for i in range(top_k):
                if sim_max[i] < 0.7:
                    continue
                j = sim_indice[i].item()
                h0, w0 = topk_indice0[i] // width0, topk_indice0[i] % width0
                h1, w1 = topk_indice1[j] // width1, topk_indice1[j] % width1
                h0_orig, w0_orig = int(h0 / scale0), int(w0 / scale0)
                h1_orig, w1_orig = int(h1 / scale1), int(w1 / scale1)
                all_hw_pairs.append([int(sim_max[i]*1000), h0_orig, w0_orig, h1_orig, w1_orig])

    return torch.tensor(all_hw_pairs, dtype=torch.int32)

# --- Draw matches ---
def draw_matches(img0, img1, hw_pairs, output_dir, base0="img0", base1="img1"):
    img0_draw = img0.copy()
    img1_draw = img1.copy()
    for score, h0, w0, h1, w1 in hw_pairs:
        cv.circle(img0_draw, (w0, h0), radius=5, color=(0, 255, 0), thickness=1)
        cv.circle(img1_draw, (w1, h1), radius=5, color=(0, 255, 0), thickness=1)
    cv.imwrite(os.path.join(output_dir, f"{base0}_matches.jpg"), img0_draw)
    cv.imwrite(os.path.join(output_dir, f"{base1}_matches.jpg"), img1_draw)
    print(f"Saved matches to {output_dir}")

# --- Main ---
if __name__ == "__main__":
    start_time = time.time()

    base_dir = "/home/jack/Desktop/archive"
    img0_path = os.path.join(base_dir, "test_image0.jpg")
    img1_path = os.path.join(base_dir, "test_image1.jpg")
    output_dir = os.path.join(base_dir, "matches_found")
    os.makedirs(output_dir, exist_ok=True)

    img0 = cv.imread(img0_path)
    img1 = cv.imread(img1_path)
    if img0 is None or img1 is None:
        raise FileNotFoundError("One of the input images could not be loaded.")

    hw_pairs = match_two_multiscale(img0, img1, scales=[1.0, 0.75, 0.5], top_k=200)

    draw_matches(img0, img1, hw_pairs, output_dir, base0="img0", base1="img1")

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    print(f"Elapsed time: {m} min {s} sec")
