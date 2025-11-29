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
    height, width = ktps.shape[1:3]
    kpts_map = torch.sigmoid(ktps).reshape(height * width)
    desc_map = desc.reshape(desc.shape[0], height * width)
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice

# --- Create image pyramid for img1 only ---
def create_image_pyramid(img, scales=[1.0]):
    pyramid = []
    for scale in scales:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv.resize(img, (new_w, new_h))
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        pyramid.append((gray, scale))
    return pyramid

# --- Match base image to pyramid of img1 ---
def match_base_to_pyramid(base_img, img_pyramid, top_k=20):
    all_hw_pairs = []

    # Prepare base image tensor once
    base_tensor = silk.utils.img_to_tensor(base_img, device=device, normalization=True)
    if base_tensor.ndim == 3:
        base_tensor = base_tensor.unsqueeze(0)
    print("Forward pass on base image...")
    base_kpts, base_desc, base_indice = get_topk(*model.forward(base_tensor), k=top_k)
    print("Base forward pass done")

    base_desc = base_desc.permute(1, 0)
    base_desc = base_desc / torch.norm(base_desc, p=2, dim=1, keepdim=True)
    h0_dim, w0_dim = base_img.shape

    for gray, scale in img_pyramid:
        print(f"Processing pyramid scale {scale:.2f}")
        img_tensor = silk.utils.img_to_tensor(gray, device=device, normalization=True)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        topk_kpts, topk_desc, topk_indice = get_topk(*model.forward(img_tensor), k=top_k)
        topk_desc = topk_desc.permute(1, 0)
        topk_desc = topk_desc / torch.norm(topk_desc, p=2, dim=1, keepdim=True)
        h1_dim, w1_dim = gray.shape

        # Similarity
        sim_mat = torch.matmul(base_desc, topk_desc.T)
        sim_max, sim_indice = torch.max(sim_mat, dim=1)

        for i in range(top_k):
            if sim_max[i] < 0.7:
                continue
            j = sim_indice[i].item()
            h0, w0 = base_indice[i] // w0_dim, base_indice[i] % w0_dim
            h1, w1 = topk_indice[j] // w1_dim, topk_indice[j] % w1_dim
            # Map to original resolution
            h1_orig = int(h1 / scale)
            w1_orig = int(w1 / scale)
            all_hw_pairs.append([int(sim_max[i]*1000), h0, w0, h1_orig, w1_orig])

        print(f"Completed matching scale {scale:.2f}")

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

    # Downscale and grayscale for VM safety
    img0 = cv.resize(img0, (320, 240))
    img0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    img1 = cv.resize(img1, (320, 240))
    img1 = cv.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Create pyramid for img1
    pyramid = create_image_pyramid(img1, scales=[1.0, 0.75, 0.5])

    hw_pairs = match_base_to_pyramid(img0, pyramid, top_k=20)

    draw_matches(img0, img1, hw_pairs, output_dir, base0="testimage0", base1="testimage1")

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    print(f"Total elapsed time: {m} min {s} sec")
