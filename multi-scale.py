import torch
import cv2 as cv
import silk
import os
import time

# --- Setup device ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load SiLK model ---
model = silk.SiLK()
model = model.to(device)
model.train(False)
model.load_state_dict(torch.load("./train0_25000.pth"))

# --- Extract top-k keypoints ---
def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=20):
    if ktps.ndim == 5:
        ktps = ktps.squeeze(0)
        desc = desc.squeeze(0)
    height, width = ktps.shape[-2:]
    kpts_map = torch.sigmoid(ktps).reshape(-1)
    desc_map = desc.reshape(desc.shape[0], -1)
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice

# --- Convert to grayscale safely ---
def to_gray(img):
    if img.ndim == 3 and img.shape[2] == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

# --- Match base image to a scaled version of img1 ---
def match_base_to_scaled(base_img, target_img, scale=1.0, top_k=20):
    # Downscale target image
    if scale != 1.0:
        h, w = target_img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        target_img_scaled = cv.resize(target_img, (new_w, new_h))
    else:
        target_img_scaled = target_img.copy()

    base_tensor = silk.utils.img_to_tensor(base_img, device=device, normalization=True)
    if base_tensor.ndim == 3:
        base_tensor = base_tensor.unsqueeze(0)
    target_tensor = silk.utils.img_to_tensor(target_img_scaled, device=device, normalization=True)
    if target_tensor.ndim == 3:
        target_tensor = target_tensor.unsqueeze(0)

    # Forward pass
    base_kpts, base_desc, base_indice = get_topk(*model.forward(base_tensor), k=top_k)
    target_kpts, target_desc, target_indice = get_topk(*model.forward(target_tensor), k=top_k)

    # Normalize descriptors
    base_desc = base_desc.permute(1, 0)
    base_desc = base_desc / torch.norm(base_desc, p=2, dim=1, keepdim=True)
    target_desc = target_desc.permute(1, 0)
    target_desc = target_desc / torch.norm(target_desc, p=2, dim=1, keepdim=True)

    h0_dim, w0_dim = base_img.shape[:2]
    h1_dim, w1_dim = target_img_scaled.shape[:2]

    # Similarity and matching
    sim_mat = torch.matmul(base_desc, target_desc.T)
    sim_max, sim_indice = torch.max(sim_mat, dim=1)

    all_hw_pairs = []
    for i in range(top_k):
        if sim_max[i] < 0.7:
            continue
        j = sim_indice[i].item()
        h0, w0 = base_indice[i] // w0_dim, base_indice[i] % w0_dim
        h1, w1 = target_indice[j] // w1_dim, target_indice[j] % w1_dim
        h1_orig = int(h1 / scale)
        w1_orig = int(w1 / scale)
        all_hw_pairs.append([int(sim_max[i]*1000), h0, w0, h1_orig, w1_orig])

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
    img0_path = os.path.join(base_dir, "testimage0.jpg")
    img1_path = os.path.join(base_dir, "testimage1.jpg")
    output_dir = os.path.join(base_dir, "matches_found")
    os.makedirs(output_dir, exist_ok=True)

    img0 = cv.imread(img0_path)
    img1 = cv.imread(img1_path)
    if img0 is None or img1 is None:
        raise FileNotFoundError("One of the input images could not be loaded.")

    # --- Multi-resolution and multi-scale testing ---
    resolutions = [320, 640, 1280]  # reference image resolutions
    scales = [0.5, 1.0, 1.5, 2.0]  # target image scale variations
    all_hw_pairs = []

    for res in resolutions:
        # Resize reference image
        h0, w0 = img0.shape[:2]
        scale_ref = res / max(h0, w0)
        img0_res = cv.resize(img0, (int(w0*scale_ref), int(h0*scale_ref)))

        for scale in scales:
            print(f"Matching resolution {res} with scale {scale}")
            pairs = match_base_to_scaled(img0_res, img1, scale=scale, top_k=20)
            if pairs.numel() > 0:
                all_hw_pairs.append(pairs)
            del pairs
            torch.cuda.empty_cache()

    hw_pairs = torch.cat(all_hw_pairs, dim=0) if all_hw_pairs else torch.empty((0,5), dtype=torch.int32)

    draw_matches(img0, img1, hw_pairs, output_dir, base0="testimage0", base1="testimage1")

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    print(f"Total elapsed time: {m} min {s} sec")
