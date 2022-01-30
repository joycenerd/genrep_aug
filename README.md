# Generative Models as a Data Augmentation for Classification
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

### [Slides](./docs/SLIDES.pdf) | [Video](https://youtu.be/y-v_K0sf_lA)

This repository is the implementation of final project for 5003 Deep Learning and Practice course in 2021 summer semester at National Yang Ming Chiao Tung University. 

In this final project we use GAN steerability as an data augmentation technique. The inspiration is coming from [GAN steerability](https://arxiv.org/pdf/1907.07171.pdf), and [GenRep](https://arxiv.org/pdf/2106.05258.pdf) these two papers. In this project, we investigate image transformation by exploring walks in the latent space of GAN. And we conclude that GAN steerability is a better data augmentation technique compare to transformation done in the data space

<p align="center">
  <img src="./figure/quality.PNG" />
</p>

## Getting the code
You can download a copy of all the files in this repository by cloning this repository:
```
git clone https://github.com/joycenerd/genrep_aug.git
```

## Requirements

You need to have [Anaconda](https://www.anaconda.com/) or Miniconda already installed in your environment. To install requirements:

```
cd GenRep
conda env create -f environment.yml
```

## GAN steer

### Train BigGAN steerability
```
cd GenRep/utils
python biggan_steer_train.py
```

### Generate augmented images
```
cd GenRep/utils
python generate_dataset_biggan_steer.py
```

## Mix data

### Choose 1300 augmented images
```
cd GenRep/utils
python extract_data.py
```

### Merge real and augmented data
```
cd GenRep/utils
python merge_real_gen.py
```

## Evaluation and Testing
**To be updated**

## Pre-trained Models

