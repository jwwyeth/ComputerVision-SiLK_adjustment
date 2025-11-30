import torch
import cv2 as cv
import silk
import os
import time
import numpy as np

# --- Setup device ---
start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load model ---
model = silk.SiLK()
model = model.to(device)
model.train(False)
model.load_state_dict(torch.load("./train0_25000.pth"))

# --- Top-k extraction ---
def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=100):
    height, width = ktps.shape[2:4]
    kpts_map = torch.sigmoid(ktps).reshape(height*width)
    desc_map = desc.reshape((-1, height*width))
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice

# --- Match two images ---
def match_two(img0_t: torch.Tensor, img1_t: torch.Tensor, threshold=0.7, K=200):
    topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0_t), k=K)
    topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1_t), k=K)
    
    topk_desc0 = topk_desc0.permute(1, 0)
    topk_desc1 = topk_desc1.permute(1, 0)
    topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
    topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)

    sim_mat = torch.matmul(topk_desc0, topk_desc1.T)
    sim_max, sim_indice = torch.max(sim_mat, dim=1)

    _, _, h0, w0 = img0_t.shape
    _, _, h1, w1 = img1_t.shape

    hw_pairs = []
    sim_scores = []
    for i in range(K):
        if sim_max[i] < threshold:
            continue
        j = sim_indice[i].item()
        hw_pairs.append([
            int(sim_max[i]*1000),
            topk_indice0[i] // w0,
            topk_indice0[i] % w0,
            topk_indice1[j] // w1,
            topk_indice1[j] % w1,
        ])
        sim_scores.append(sim_max[i].item())
    return torch.tensor(hw_pairs, dtype=torch.int32), sim_scores

# --- Image directories ---
base_dir = os.path.join("..", "archive")
images_dir = os.path.join(base_dir, "images_test")
output_dir = os.path.join(base_dir, "match_test")
os.makedirs(output_dir, exist_ok=True)

image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".jpg")]

# --- Process each image ---
for img_path in image_paths:
    print("="*60)
    if not os.path.exists(img_path):
        print(f"Not found: {img_path}")
        continue
    img = cv.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        continue

    img0 = img.copy()
    scale_factor = 0.5
    new_w = int(img0.shape[1]*scale_factor)
    new_h = int(img0.shape[0]*scale_factor)
    img1 = cv.resize(img0, (new_w, new_h))

    img0_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img0_tensor = silk.utils.img_to_tensor(img0_gray, device=device, normalization=True)
    img1_tensor = silk.utils.img_to_tensor(img1_gray, device=device, normalization=True)

    K = 200
    hw_pairs, sim_scores = match_two(img0_tensor, img1_tensor, threshold=0.7, K=K)

    mean_similarity = sum(sim_scores)/len(sim_scores) if len(sim_scores)>0 else 0
    matching_score = len(sim_scores)/K

    # --- Visualization ---
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1

    for _, y0, x0, y1, x1 in hw_pairs:
        pt0 = (int(x0), int(y0))
        pt1 = (int(x1)+w0, int(y1))
        cv.circle(canvas, pt0, 2, (0,0,255), -1)
        cv.circle(canvas, pt1, 2, (0,0,255), -1)
        cv.line(canvas, pt0, pt1, (0,255,0), 1)

    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(output_dir, f"{name}_scale_match_test{ext}")
    cv.imwrite(save_path, canvas)
    print(f"Matches found: {len(hw_pairs)} | Saved to: {save_path}")
    print("="*60)

elapsed = time.time() - start_time
m, s = divmod(int(elapsed), 60)
print("\n" + "="*60)
print("STATISTICS SUMMARY")
print("="*60)
print(f"Matches: {len(hw_pairs)}")
print(f"Mean similarity: {mean_similarity:.4f}")
print(f"Matching score: {matching_score:.4f}")
print(f"Total execution time: {m} min {s} sec")
print("="*60)
