import torch
import cv2 as cv
import numpy as np
import silk
import os
import time

print("Testing...")
start_time = time.time()

device = None
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
    device = torch.device("cuda:0")
else:
    print("CUDA is not available. Training on CPU.")
    device = torch.device("cpu")

model = silk.SiLK()
model = model.to(device)
model.train(False)
model.load_state_dict(torch.load("./train0_25000.pth"))


def get_topk(ktps: torch.Tensor, desc: torch.Tensor, k=100):
    height, width = ktps.shape[2:4]
    kpts_map = torch.sigmoid(ktps).reshape(height*width)
    desc_map = desc.reshape((-1, height*width))
    topk_value, topk_indice = kpts_map.topk(k)
    topk_desc = desc_map[:, topk_indice]
    return topk_value, topk_desc, topk_indice


def match_two(img0: torch.Tensor, img1: torch.Tensor):
    """this function can match two images.

    Args:
        img0 (torch.Tensor): shaped of [1, 1, H, W], formed by `utils.img_to_tensor`
        img1 (torch.Tensor): shaped of [1, 1, H, W], formed by `utils.img_to_tensor`
    """
    height, width = img0.shape[2:4]
    K = 200
    topk_kpts0, topk_desc0, topk_indice0 = get_topk(*model.forward(img0), k=K)
    topk_kpts1, topk_desc1, topk_indice1 = get_topk(*model.forward(img1), k=K)
    topk_desc0 = topk_desc0.permute(1, 0)  # shape = [K, 128]
    topk_desc1 = topk_desc1.permute(1, 0)  # shape = [K, 128]
    topk_desc0 = topk_desc0 / torch.norm(topk_desc0, p=2, dim=1, keepdim=True)
    topk_desc1 = topk_desc1 / torch.norm(topk_desc1, p=2, dim=1, keepdim=True)
    sim_mat = torch.matmul(topk_desc0, topk_desc1.T)  # Similarity from desc0 to desc1
    sim_max, sim_indice = torch.max(sim_mat, dim=1)
    hw_pairs = []
    for i in range(K):
        if sim_max[i] < 0.7:
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

# Paths
base_dir = os.path.join("..", "archive")
transformed_dir = os.path.join(base_dir, "transformed")
coco_imgs_dir = os.path.join(base_dir, "images")


output_dir = "matches_found"
os.makedirs(output_dir, exist_ok=True)

# Load images
images = []
for f in os.listdir(coco_imgs_dir):
    if f.endswith(".jpg"):
        images.append(os.path.join(coco_imgs_dir, f))

images_test = images[:5]

scaled_imgs = []
for f in os.listdir(transformed_dir):
    if f.endswith(".jpg"):
        scaled_imgs.append(os.path.join(transformed_dir, f))


# Convert images to tensors
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
for path in images:
    img_cv = cv.imread(path)
    img_gray = cv.cvtColor(cv.resize(img_cv, (160, 120)), cv.COLOR_BGR2GRAY)
    img_tensor = silk.utils.img_to_tensor(img_gray, device=device, normalization=True)
    coco_images_tensor.append(img_tensor)
    coco_images_cv.append(img_cv)


# Match images
for i in range(len(scaled_images_tensor)):
    img0_tensor = scaled_images_tensor[i]
    img1_tensor = coco_images_tensor[i]
    img0_cv = images_cv[i]
    img1_cv = coco_images_cv[i]

    hw_pairs = match_two(img0_tensor, img1_tensor)

    # Draw matches
    img0_draw = img0_cv.copy()
    img1_draw = img1_cv.copy()
    for score, h0, w0, h1, w1 in hw_pairs:
        cv.circle(img0_draw, (w0.item(), h0.item()), radius=5, color=(0, 255, 0), thickness=1)
        cv.circle(img1_draw, (w1.item(), h1.item()), radius=5, color=(0, 255, 0), thickness=1)

    # Save images
    base0 = os.path.basename(scaled_imgs[i])
    base1 = os.path.basename(images[i])
    cv.imwrite(os.path.join(output_dir, f"{base0}_matches.jpg"), img0_draw)
    cv.imwrite(os.path.join(output_dir, f"{base1}_matches.jpg"), img1_draw)
    #print(f"Saved matches between {base0} and {base1} to {output_dir}")

end_time = time.time()
elapsed = end_time - start_time
m = int(elapsed // 60)
s = int(elapsed % 60)
print(f"Elapsed Time: {m} min {s} sec")
# Test 1: 222 min 1 sec