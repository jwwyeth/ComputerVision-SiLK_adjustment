import torch
import cv2 as cv
import silk
import os
import time
import numpy as np

# --- Setup device ---
start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = silk.SiLK()
model = model.to(device)
model.train(False)
model.load_state_dict(torch.load("./train0_25000.pth"))

# --- Top-k extraction (Unchanged) ---
def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=100):
    height, width = ktps.shape[2:4]
    kpts_map = torch.sigmoid(ktps).reshape(height*width)
    desc_map = desc.reshape((-1, height*width))
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice

# --- Match two images ---
def match_two(img0_t: torch.Tensor, img1_t: torch.Tensor, threshold=0.7, K=200):
    # Get features
    topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0_t), k=K)
    topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1_t), k=K)
    
    # Normalize descriptors
    topk_desc0 = topk_desc0.permute(1, 0)
    topk_desc1 = topk_desc1.permute(1, 0)
    topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
    topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)

    # Match
    sim_mat = torch.matmul(topk_desc0, topk_desc1.T)
    sim_max, sim_indice = torch.max(sim_mat, dim=1)

    # Get Dimensions of both images to decode indices correctly
    _, _, h0, w0 = img0_t.shape
    _, _, h1, w1 = img1_t.shape

    hw_pairs = []
    matched_sims = []
    for i in range(K):
        if sim_max[i] < threshold:
            continue
        j = sim_indice[i].item()
        
        # We must use w0 for image 0 and w1 for image 1
        hw_pairs.append([
            int(sim_max[i] * 1000),
            topk_indice0[i] // w0,  # y0
            topk_indice0[i] % w0,   # x0
            topk_indice1[j] // w1,  # y1
            topk_indice1[j] % w1,   # x1
        ])
        matched_sims.append(sim_max[i].item())
    return torch.tensor(hw_pairs, dtype=torch.int32), matched_sims

# --- Load and Scale Images ---
base_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(base_dir, "images")
output_dir = os.path.join(base_dir, "match_test")
os.makedirs(output_dir, exist_ok=True)

image_paths = []
for f in os.listdir(images_dir):
    if f.endswith(".jpg"):
        full = os.path.join(base_dir, f"images/{f}")
        image_paths.append(os.path.abspath(full))

for img_path in image_paths:
    print("=" * 70)
    if not os.path.exists(img_path):
        print(f"Not found, skipping: {img_path}")
        continue
    
    img = cv.imread(img_path)
    if img is None:
        print(f"Failed to load, skipping: {img_path}")
        continue

    original_img = cv.imread(img_path)

    # Prepare Image 0 (original Size)
    img0 = original_img.copy()

    # Prepare Image 1 (scaled)
    scale_factor = 0.5 # ---------------- CHANGE ----------------
    new_width = int(img0.shape[1] * scale_factor)
    new_height = int(img0.shape[0] * scale_factor)
    img1 = cv.resize(img0, (new_width, new_height))

    print(f"Img0 Shape: {img0.shape} | Img1 Shape: {img1.shape} (Scale: {scale_factor})")

    img0_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    img0_tensor = silk.utils.img_to_tensor(img0_gray, device=device, normalization=True)
    img1_tensor = silk.utils.img_to_tensor(img1_gray, device=device, normalization=True)

    # --- Run Matching ---
    K = 200
    hw_pairs, sim_scores = match_two(img0_tensor, img1_tensor, threshold=0.7, K=K)

    if len(sim_scores) > 0:
        mean_similarity = sum(sim_scores) / len(sim_scores)
        matching_score = len(sim_scores) / K 
    else:
        mean_similarity = 0
        matching_score = 0

    # --- Visualize Side-by-Side ---
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place images on canvas
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1

    # Draw lines
    for _, y0, x0, y1, x1 in hw_pairs:
        pt0 = (int(x0), int(y0))
        # Shift pt1 by width of img0 so it appears on the second image
        pt1 = (int(x1) + w0, int(y1)) 
        
        # Draw line and keypoints
        cv.line(canvas, pt0, pt1, (0, 255, 0), 1)
        cv.circle(canvas, pt0, 1, (0, 0, 255), -1)
        cv.circle(canvas, pt1, 1, (0, 0, 255), -1)

    # Save new image
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)

    new_filename = f"{name}_scale_match_test{ext}"  
    save_path = os.path.join(output_dir, new_filename)
    cv.imwrite(save_path, canvas)

    print(f"Matches found for {filename}: {len(hw_pairs)}")
    print("=" * 70)

elapsed = time.time() - start_time
m, s = divmod(int(elapsed), 60)

print("\n" + "=" * 70)
print("STATISTICS SUMMARY")
print("=" * 70)
print(f"Matches: {len(hw_pairs)}")
print(f"Mean similarity: {mean_similarity:.4f}")
print(f"Matching score: {matching_score:.4f}")

print(f"Total execution time: {m} min {s} sec")
print("=" * 70)