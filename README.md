# Testing SiLK at different scales

## Dataset

We used COCO minitrain2017 database to test.
Link:
https://www.kaggle.com/datasets/trungit/coco25k


## How to Run

Before running this, ensure that you have the correct scale that you want to test for is changed and that you have your images in the images folder.

```
pip install torch opencv-python numpy

python test_match.py
```

##File Explanation

Base_Silk_output: The output pictures when running the base SiLK-Lite program
Scaled_otput: The output pictures when running the scaled SiLK-Lite program
images: THe input image used for testing.
silk: The reference folder from SiLK-Lite that referneces all tools of the SiLK program that are referenced in base and scaled matching scripts.
Base_version_test_match.py: The orignial version of SiLK that detects keypoints between a base image and target image of the same scale.
coco_minitrain2017.csv:  A cvs file of the coco_minitrain2017 dataset which was used to train our SiLK model on.
test_match_scaled.py:The agumented  version of SiLK that detects keypoints between a base image and target image of the differing scale.
train.py: The program that trains a SiLK model in refernece to a given dataset such as coco_minitrain2017.
train0_25000.pth: The produced trained model after running train.py using coco_minitrain2017 composed of 25,000 images.




