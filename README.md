# Exploring Multimodal Large Language Models for Medical Image Captioning

This repository contains the research and code for the thesis titled **"Exploring Multimodal Large Language Models for Medical Image Captioning."** The goal of this work is to investigate the potential of Multimodal Large Language Models (MLLMs) in the Diagnostic Captioning task. This research focuses on developing techniques that combine both visual and textual information to enhance the performance of automatic captioning systems for medical image analysis.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [License](#license)
- [Citation](#citation)

## Introduction

This research focuses on utilizing **Multimodal Large Language Models (MLLMs)** for medical image captioning tasks. The objective is to generate precise and descriptive captions for medical images (e.g., X-rays, CT scans, MRI scans) to aid healthcare professionals in diagnostic decision-making.

This repository provides the following resources:

- **Fine-tuning scripts** for adapting LlaVA 1.5 on a dataset.
- **Few-shot adaptation scripts** for LlaVA 1.5,utilizing image-caption pairs retrieved from the k-nearest neighbors of each test image in the training set.
- Code for the **MLLM Synthesizer**, which integrates information from similar images and their corresponding neighbors to generate captions for test images.
- Code for **LM-Fuser**, which aggregates captions from various MLLMs using their pretrained weights and trains a smaller language model solely on these outputs.


## Dataset

The dataset used in this research was provided by the [ImageCLEF 2024 competition](https://www.imageclef.org/2024).

It consists of 80,080 image-caption pairs, along with associated image concepts.
The dataset includes two tasks: **Concept Detection** and **Caption Prediction**. This research focuses on the latter task.

## Models Performance Comparison

| Model                | BertScore | ROUGE-L | BLEURT | BLEU  |
|----------------------|-----------|---------|--------|-------|
| LLaVA 1.5 SFT        |   62.08   |   22.53 |  30.99 |  20.8 |
| LLaVA 1.5 Few-Shot   |   59.02   |   20.60 |  28.25 |  17.6 |
| Llama 3.1 Synthesizer|   63.55   |   23.29 |  31.44 |  21.7 |
| LM-Fuser             |   64.27   |   24.40 |  31.99 |  21.4 |


# License
This repository is licensed under the MIT license. See [LICENSE](LICENSE) for more details.

## Citation

If you use this repository or parts of this work in your own research, please cite the following:

```
@mastersthesis{samprovalaki2024multimodal,
  author       = {M. Samprovalaki},
  title        = {Exploring Multimodal Large Language Models for Medical Image Captioning},
  school       = {Athens University of Economics and Business, Department of Informatics},
  year         = {2024},
}
```
