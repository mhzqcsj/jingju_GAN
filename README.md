# jingju_GAN
Generate facial makeup

````markdown
# Peking Opera Facial Makeup Pattern Generation

## Overview

This repository contains the dataset, code, and experimental results for the research project on **High-Quality Generation of Traditional Peking Opera Facial Makeup Patterns Based on Region-Aware and Multi-Attention Mechanisms**.

The project addresses challenges in generating realistic and diverse Peking Opera facial makeup patterns by proposing a novel generation framework that integrates an SDXL-based pattern augmentation workflow and an improved StyleGAN model enhanced with self-attention and border-attention mechanisms. A dedicated image-to-image (img2img) pipeline is further applied for refined high-fidelity pattern generation.



## Dataset

The dataset, named **Jingjushow**, includes 640 high-quality Peking Opera facial makeup images covering diverse styles and regions.
Images and annotations are carefully preprocessed and labeled with key prompt words to facilitate pattern generation.
Dataset and resources can be downloaded from:

  [Baidu Pan Link](https://pan.baidu.com/s/1tQisqApVh5hDUeQ7YvglJQ?pwd=vkpe)  
  Extraction Code: `vkpe`

The dataset is also hosted on GitHub:  
  https://github.com/mhzqcsj/jingju_GAN



## Abstract

Traditional Peking Opera facial makeup patterns carry rich cultural meanings and intricate visual features. This work proposes a high-quality generation method that integrates region-aware segmentation and multi-attention mechanisms to generate faithful and artistically consistent facial makeup patterns. The method combines an SDXL-based data augmentation workflow and an improved StyleGAN model with self-attention and border-attention modules, followed by a refined image-to-image processing stage. Extensive experiments demonstrate that the proposed framework achieves superior performance in structural fidelity, visual clarity, and style consistency over existing methods.



## Keywords

 Peking Opera Facial Makeup  
 Pattern Generation  
 Region-Aware Segmentation  
 Attention Mechanisms  
 Generative Adversarial Networks



## Experimental Results

 The improved StyleGAN model achieves the lowest FID score of 43.72 on template pattern generation, outperforming baseline models.
 The image-to-image workflow integrated with IP-Adapter yields the best refined pattern generation results with FID=38.72, SSIM=0.57, and PSNR=19.46.
 The dataset augmentation increases diversity (LPIPS=0.36) while maintaining close distribution similarity with original data (FID=45.12, SSIM=0.57).
 Visual examples and quantitative comparisons are provided in the repository for reproduction and evaluation.



### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mhzqcsj/jingju_GAN.git
   cd jingju_GAN
````

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Recommended environment:

   * Python 3.8+
   * PyTorch 1.8.1+
   * CUDA 11.3+
   * NVIDIA RTX 3090 Ti or equivalent GPU

### Data Preprocessing

* Run the preprocessing script to normalize images and generate annotations:

  ```bash
  python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
  ```

### Training

* Train the improved StyleGAN model on the Jingjushow dataset:

  ```bash
  python train.py --config configs/stylegan_improved.yaml
  ```

### Inference and Refinement

* Generate Peking Opera facial makeup patterns and perform refinement with the img2img pipeline:

  ```bash
  python inference.py --model_path checkpoints/stylegan_improved.pth --refine True
  ```
