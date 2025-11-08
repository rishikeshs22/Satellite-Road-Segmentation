# 🛰️ UNet-Based Semantic Segmentation of Satellite Imagery for Geospatial Feature Extraction


This project implements a UNet model using PyTorch for segmenting satellite images. The model is trained on paired image-mask samples and deployed using a Gradio interface via Hugging Face Spaces.

This project was done by Harshit, Abhishek, Vishal and Rishikesh as part of the AI/ML course for Designers under the guidance of Prof. Amar Behera. We would like to express our sincere appreciation to Amar Sir for designing such a wonderful course, which provided us with valuable exposure to real-world machine learning workflows, hands-on model implementation, and practical problem-solving in AI.

## 🌐 Live Demo

👉 [Check out the app on Hugging Face Spaces](https://huggingface.co/spaces/rishikeshs22/Satellite-Road-Segmentation) 

---

## 📁 Files in This Repository

- `app.py`: Gradio app to run inference using the trained UNet model.
- `train.py`: Script to train the UNet model on image-mask pairs.
- `model.py`: PyTorch implementation of the UNet architecture.
- `dataset.py`: Custom dataset loader for segmentation data.
- `requirements.txt`: Python dependencies for running the app.

---
## 🔍 Sample training datas

Here are some examples from the training set showing input images, ground truth masks.

| Satellite Image | Ground Truth Mask |
|-----------------|-------------------|
| ![](Examples/img_1.png) | ![](Examples/ex_mask_1.png) |
| ![](Examples/img_2.png) | ![](Examples/ex_mask_2.png) |
| ![](Examples/img_3.png) | ![](Examples/ex_mask_3.png) |

## 🔍 Sample Result

Here are result using our model.

| Satellite Image | Ground Truth Mask |
|-----------------|-------------------|
| ![](Examples/test_1.png) | ![](Examples/result.webp) |

