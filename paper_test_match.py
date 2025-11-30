import torch
import cv2 as cv
import silk
import os
import time

# --- Setup device ---
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
def match_two(img0: torch.Tensor, img1: torch.Tensor, threshold=0.7, K=200):
    topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0), k=K)
    topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1), k=K)
    topk_desc0 = topk_desc0.permute(1, 0)
    topk_desc1 = topk_desc1.permute(1, 0)
    topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
    topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)

    sim_mat = torch.matmul(topk_desc0, topk_desc1.T)
    sim_max, sim_indice = torch.max(sim_mat, dim=1)

    height, width = img0.shape[2:4]
    hw_pairs = []
    for i in range(K):
        if sim_max[i] < threshold:
            continue
        j = sim_indice[i].item()
        hw_pairs.append([
            int(sim_max[i] * 1000),
            topk_indice0[i] // width,
            topk_indice0[i] % width,
            topk_indice1[j] // width,
            topk_indice1[j] % width,
        ])
    return torch.tensor(hw_pairs, dtype=torch.int32)

# --- Load and preprocess images ---
img0_path = "/home/jack/Desktop/archive/testimage0.jpg"
img1_path = "/home/jack/Desktop/archive/testimage1_ud.jpg"
img0 = cv.imread(img0_path)
img1 = cv.imread(img1_path)
img0 = cv.resize(img0, (160, 120))
img1 = cv.resize(img1, (160, 120))
img0_gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img0_tensor = silk.utils.img_to_tensor(img0_gray, device=device, normalization=True)
img1_tensor = silk.utils.img_to_tensor(img1_gray, device=device, normalization=True)

# --- Measure forward pass time ---
start_time = time.time()
hw_pairs = match_two(img0_tensor, img1_tensor)
elapsed_time = time.time() - start_time

# --- Metrics ---
K = 200
num_matches = hw_pairs.shape[0]
matching_score = num_matches / K if K > 0 else 0
mean_similarity = (hw_pairs[:,0].float() / 1000).mean().item() if num_matches > 0 else 0

print(f"Number of matches: {num_matches}")
print(f"Matching score: {matching_score:.3f}")
print(f"Mean similarity: {mean_similarity:.3f}")
print(f"Total computation time: {elapsed_time:.2f} seconds")

# --- Draw matches ---
output_dir = "/home/jack/Desktop/matches_found"
os.makedirs(output_dir, exist_ok=True)
img0_draw = img0.copy()
img1_draw = img1.copy()
for _, h0, w0, h1, w1 in hw_pairs:
    cv.circle(img0_draw, (w0.item(), h0.item()), radius=5, color=(0, 255, 0), thickness=1)
    cv.circle(img1_draw, (w1.item(), h1.item()), radius=5, color=(0, 255, 0), thickness=1)

cv.imwrite(os.path.join(output_dir, "img0_matches.jpg"), img0_draw)
cv.imwrite(os.path.join(output_dir, "img1_matches.jpg"), img1_draw)
print(f"Saved matched images to folder: {output_dir}")
