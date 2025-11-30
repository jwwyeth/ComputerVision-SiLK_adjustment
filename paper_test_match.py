import torch
import cv2 as cv
import numpy as np
import silk
import os

# --- Device setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load SiLK model ---
model = silk.SiLK()
model = model.to(device)
model.eval()
model.load_state_dict(torch.load("./train0_25000.pth"))

# --- Utility functions ---
def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=200):
    height, width = ktps.shape[2:4]
    kpts_map = torch.sigmoid(ktps).reshape(height*width)
    desc_map = desc.reshape((-1, height*width))
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice

def match_two(img0_tensor: torch.Tensor, img1_tensor: torch.Tensor, sim_thresh=0.7):
    height, width = img0_tensor.shape[2:4]
    K = 200
    topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0_tensor), k=K)
    topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1_tensor), k=K)

    topk_desc0 = topk_desc0.permute(1,0)
    topk_desc1 = topk_desc1.permute(1,0)

    topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
    topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)

    sim_mat = torch.matmul(topk_desc0, topk_desc1.T)
    sim_max, sim_indice = torch.max(sim_mat, dim=1)

    hw_pairs = []
    sim_scores = []

    for i in range(K):
        if sim_max[i] < sim_thresh:
            continue
        j = sim_indice[i].item()
        hw_pairs.append([
            int(sim_max[i]*1000),
            topk_indice0[i] // width,
            topk_indice0[i] % width,
            topk_indice1[j] // width,
            topk_indice1[j] % width,
        ])
        sim_scores.append(sim_max[i].item())

    return torch.tensor(hw_pairs, dtype=torch.int32), sim_scores

def prep_image_tensor(img_path, resize_shape=None):
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    if resize_shape is not None:
        img = cv.resize(img, resize_shape)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    tensor = silk.utils.img_to_tensor(img_gray, device=device, normalization=True)
    return img, tensor

def draw_matches(img0, img1, hw_pairs, output_dir, base0="img0", base1="img1"):
    img0_draw = img0.copy()
    img1_draw = img1.copy()
    hw_pairs = hw_pairs.cpu().numpy() if isinstance(hw_pairs, torch.Tensor) else hw_pairs

    for _, h0, w0, h1, w1 in hw_pairs:
        cv.circle(img0_draw, (w0, h0), radius=5, color=(0,255,0), thickness=1)
        cv.circle(img1_draw, (w1, h1), radius=5, color=(0,255,0), thickness=1)

    os.makedirs(output_dir, exist_ok=True)
    cv.imwrite(os.path.join(output_dir, f"{base0}_matches.jpg"), img0_draw)
    cv.imwrite(os.path.join(output_dir, f"{base1}_matches.jpg"), img1_draw)
    print(f"Saved matched images to {output_dir}")

# --- Main ---
img0_path = "/home/jack/Desktop/testimage0.jpg"
img1_path = "/home/jack/Desktop/testimage1.jpg"
output_dir = "/home/jack/Desktop/matches_found"

img0, img0_tensor = prep_image_tensor(img0_path)
img1, img1_tensor = prep_image_tensor(img1_path)

hw_pairs, sim_scores = match_two(img0_tensor, img1_tensor)

num_matches = len(hw_pairs)
matching_score = sum([score for score, _,_,_,_ in hw_pairs])/1000 / num_matches if num_matches > 0 else 0
mean_similarity = np.mean(sim_scores) if sim_scores else 0.0

print(f"Number of matches: {num_matches}")
print(f"Matching score: {matching_score:.3f}")
print(f"Mean similarity: {mean_similarity:.3f}")

# Histogram of similarity scores
import matplotlib.pyplot as plt
if sim_scores:
    plt.hist(sim_scores, bins=20, range=(0.7,1.0))
    plt.title("Histogram of Cosine Similarity Scores")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir,"similarity_histogram.png"))
    plt.close()
    print(f"Saved similarity histogram to {output_dir}")

draw_matches(img0, img1, hw_pairs, output_dir)
