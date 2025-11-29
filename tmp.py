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

# --- Adjusted get_topk ---
def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=100):
    height, width = ktps.shape[2:4]

    # --- FIX 1: Correct flattening ---
    kpts_map = torch.sigmoid(ktps).squeeze(0).squeeze(0)  # [H, W]
    kpts_map = kpts_map.reshape(height * width)

    desc_map = desc.squeeze(0).reshape(desc.shape[1], height * width)
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]

    return topk_value, topk_desc, topk_indice

# --- Adjusted match_two ---
def match_two(img0: torch.Tensor, img1: torch.Tensor):
    height, width = img0.shape[2:4]
    K = 200

    topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0), k=K)
    topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1), k=K)

    topk_desc0 = topk_desc0.permute(1, 0)
    topk_desc1 = topk_desc1.permute(1, 0)

    topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
    topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)

    sim_mat = torch.matmul(topk_desc0, topk_desc1.T)
    sim_max, sim_indice = torch.max(sim_mat, dim=1)

    hw_pairs = []
    for i in range(K):
        if sim_max[i] < 0.7:
            continue
        j = sim_indice[i].item()
        hw_pairs.append([
            int(sim_max[i] * 1000),
            topk_indice0[i] // width,  # row
            topk_indice0[i] % width,   # col
            topk_indice1[j] // width,
            topk_indice1[j] % width,
        ])

    return torch.tensor(hw_pairs, dtype=torch.int32)

# --- Draw matches with scaling back to original image size ---
def draw_matches(scaled_img_cv, orig_img_cv, hw_pairs):
    # --- FIX 2: Scale coordinates back to original image size ---
    scale_x = orig_img_cv.shape[1] / 160
    scale_y = orig_img_cv.shape[0] / 120

    img_draw = orig_img_cv.copy()
    for score, h0, w0, h1, w1 in hw_pairs:
        x_up = int(w0.item() * scale_x)
        y_up = int(h0.item() * scale_y)
        cv.circle(img_draw, (x_up, y_up), radius=5, color=(0, 255, 0), thickness=1)
    return img_draw

# --- Paths ---
base_dir = os.path.join("..", "archive")
transformed_dir = os.path.join(base_dir, "transformed")
coco_imgs_dir = os.path.join(base_dir, "images")
output_dir = os.path.join("..", "matches_found")
os.makedirs(output_dir, exist_ok=True)

# --- Load images ---
scaled_imgs = [os.path.join(transformed_dir, f) for f in os.listdir(transformed_dir) if f.endswith(".jpg")]
coco_imgs = [os.path.join(coco_imgs_dir, f) for f in os.listdir(coco_imgs_dir) if f.endswith(".jpg")]

# --- Convert images to tensors ---
scaled_images_tensor = []
images_cv = []
for path in scaled_imgs:
    img_cv = cv.imread(path)
    img_gray = cv.cvtColor(cv.resize(img_cv, (160, 120)), cv.COLOR_BGR2GRAY)
    img_tensor = silk.utils.img_to_tensor(img_gray, device=device, normalization=True)
    scaled_images_tensor.append(img_tensor)
    images_cv.append(img_cv)

coco_images_tensor = []
coco_images_cv = []
for path in coco_imgs:
    img_cv = cv.imread(path)
    img_gray = cv.cvtColor(cv.resize(img_cv, (160, 120)), cv.COLOR_BGR2GRAY)
    img_tensor = silk.utils.img_to_tensor(img_gray, device=device, normalization=True)
    coco_images_tensor.append(img_tensor)
    coco_images_cv.append(img_cv)

# --- Start timing ---
start_time = time.time()

# --- Match images and save ---
for i in range(len(scaled_images_tensor)):
    hw_pairs = match_two(scaled_images_tensor[i], coco_images_tensor[i])

    img0_draw = draw_matches(scaled_images_tensor[i], images_cv[i], hw_pairs)
    img1_draw = draw_matches(coco_images_tensor[i], coco_images_cv[i], hw_pairs)

    base0 = os.path.basename(scaled_imgs[i])
    base1 = os.path.basename(coco_imgs[i])
    cv.imwrite(os.path.join(output_dir, f"{base0}_matches.jpg"), img0_draw)
    cv.imwrite(os.path.join(output_dir, f"{base1}_matches.jpg"), img1_draw)

# --- End timing ---
end_time = time.time()
elapsed = end_time - start_time
m = int(elapsed // 60)
s = int(elapsed % 60)
print(f"Elapsed Time: {m} min {s} sec")
