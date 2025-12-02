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

## File Explanation
* **Base_Silk_output**: The output pictures when running the base SiLK-Lite program
* **Scaled_output**: The output pictures when running the scaled SiLK-Lite program
* **images**: The input image used for testing
* **silk**: The reference folder from SiLK-Lite that references all tools of the SiLK program used in the base and scaled matching scripts
* **Base_version_test_match.py**: The original version of SiLK that detects keypoints between a base image and a target image of the same scale
* **coco_minitrain2017.csv**: A CSV file from the COCO MiniTrain2017 dataset used to train the SiLK model
* **test_match_scaled.py**: The augmented version of SiLK that detects keypoints between a base image and a target image of differing scales
* **train.py**: The program that trains a SiLK model using a given dataset such as `coco_minitrain2017`
* **train0_25000.pth**: The trained model produced after running `train.py` on 25,000 images from `coco_minitrain2017`




