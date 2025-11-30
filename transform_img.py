import os
import time
import numpy as np
import cv2


print("Transforming images...")
start_time = time.time()
s = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])

base_dir = os.path.join("..", "archive")
images_dir = os.path.join(base_dir, "images")

output_dir = os.path.join(base_dir, "transformed")
os.makedirs(output_dir, exist_ok=True)


image_paths = []
for f in os.listdir(images_dir):
    if f.endswith(".jpg"):
        full = os.path.join(base_dir, f"images/{f}")
        image_paths.append(os.path.abspath(full))

# with open(txt_file, "r") as f:
#     for line in f:
#         img_rel = line.strip()
#         img_rel = img_rel.lstrip("./")  # remove leading ./
#         full = os.path.join(base_dir, img_rel)
#         image_paths.append(os.path.abspath(full))

# test print
# for p in imgs_path[:5]:
#     print(p)

# Test images
image_paths_test = image_paths[:5]

for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"Not found, skipping: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load, skipping: {img_path}")
        continue

    h, w = img.shape[:2]

    # Apply transform with warpPerspective
    scaled = cv2.warpPerspective(img, s, (w, h))

    # Convert to grayscale
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # Save new image
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_scaled{ext}"  
    save_path = os.path.join(output_dir, new_filename)
    cv2.imwrite(save_path, gray)

    #print(f"Saved: {save_path}")

end_time = time.time()
elapsed = end_time - start_time

print("--- Done! ---")
print(f"It took {elapsed} s")
