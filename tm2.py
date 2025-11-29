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
def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=100):
    """Extract top-k keypoints and descriptors from SiLK output."""
    if ktps.ndim == 5:  # adjust for extra batch dim
        ktps = ktps.squeeze(0)
        desc = desc.squeeze(0)
    height, width = ktps.shape[1:3]
    kpts_map = torch.sigmoid(ktps).reshape(height * width)
    desc_map = desc.reshape(desc.shape[0], height * width)
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice

# --- Create multi-scale image pyramid ---
def create_image_pyramid(img, scales=[1.0, 0.75, 0.5]):
    pyramid = []
    for scale in scales:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv.resize(img, (new_w, new_h))
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        pyramid.append((gray, scale))
    return pyramid

# --- Multi-scale matching with timers ---
def match_two_multiscale(img0, img1, scales=[1.0, 0.75, 0.5], top_k=200):
    pyramid0 = create_image_pyramid(img0, scales)
    pyramid1 = create_image_pyramid(img1, scales)
    all_hw_pairs = []

    for s0_idx, (gray0, scale0) in enumerate(pyramid0):
        t0_scale0 = time.time()
        img0_tensor = silk.utils.img_to_tensor(gray0, device=device, normalization=True)
        if img0_tensor.ndim == 3:
            img0_tensor = img0_tensor.unsqueeze(0)

        for s1_idx, (gray1, scale1) in enumerate(pyramid1):
            t0_scale1 = time.time()
            img1_tensor = silk.utils.img_to_tensor(gray1, device=device, normalization=True)
            if img1_tensor.ndim == 3:
                img1_tensor = img1_tensor.unsqueeze(0)

            t_forward0 = time.time()
            topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0_tensor), k=top_k)
            print(f"Forward pass img0 scale {scale0:.2f} took {time.time() - t_forward0:.2f}s")

            t_forward1 = time.time()
            topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1_tensor), k=top_k)
            print(f"Forward pass img1 scale {scale1:.2f} took {time.time() - t_forward1:.2f}s")

            # Normalize descriptors
            t_norm = time.time()
            topk_desc0 = topk_desc0.permute(1, 0)
            topk_desc1 = topk_desc1.permute(1, 0)
            topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
            topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)
            print(f"Descriptor normalization took {time.time() - t_norm:.2f}s")

            # Similarity and matching
            t_match = time.time()
            sim_mat = torch.matmul(topk_desc0, topk_desc1.T)
            sim_max, sim_indice = torch.max(sim_mat, dim=1)

            h0_dim, w0_dim = gray0.shape
            h1_dim, w1_dim = gray1.shape

            for i in range(top_k):
                if sim_max[i] < 0.7:
                    continue
                j = sim_indice[i].item()
                h0, w0 = topk_indice0[i] // w0_dim, topk_indice0[i] % w0_dim
                h1, w1 = topk_indice1[j] // w1_dim, topk_indice1[j] % w1_dim
                h0_orig = int(h0 / scale0)
                w0_orig = int(w0 / scale0)
                h1_orig = int(h1 / scale1)
                w1_orig = int(w1 / scale1)
                all_hw_pairs.append([int(sim_max[i]*1000), h0_orig, w0_orig, h1_orig, w1_orig])
            print(f"Matching scale img0:{scale0:.2f} img1:{scale1:.2f} took {time.time() - t_match:.2f}s")
        print(f"Finished all img1 scales for img0 scale {scale0:.2f} in {time.time() - t0_scale0:.2f}s")

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

    hw_pairs = match_two_multiscale(img0, img1, scales=[1.0, 0.75, 0.5], top_k=200)

    draw_matches(img0, img1, hw_pairs, output_dir, base0="testimage0", base1="testimage1")

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    print(f"Total elapsed time: {m} min {s} sec")
