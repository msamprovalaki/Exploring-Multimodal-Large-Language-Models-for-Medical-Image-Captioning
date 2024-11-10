# Exploring Multimodal Large Language Models for Medical Image Captioning

This repository contains the research and code for the thesis titled **"Exploring Multimodal Large Language Models for Medical Image Captioning."** The goal of this work is to investigate the potential of Multimodal Large Language Models (MLLMs) in the Diagnostic Captioning task. This research focuses on developing techniques that combine both visual and textual information to enhance the performance of automatic captioning systems for medical image analysis.

## Table of Contents

- [Introduction](#introduction)
- [Experiments](#experiments)
- [Conclusion](#conclusion)
- [License](#license)
- [Citation](#citation)

## Introduction

In this research, we explore the Multimodal Large Language Models (MLLMs) in the medical image captioning tasks. The aim is to generate accurate and descriptive captions for medical images (e.g., X-rays, CT scans, MRI scans) to assist healthcare professionals in diagnostic processes.

This repository includes:
- Preprocessing scripts for medical image data.
- Code for fine-tuning LlaVA 1.5 on the ImageCLEF dataset.
- Code for MLLM Synthesizer that integrates information from similar images and their corresponding neighbors in order to create a caption for the test image.
- Code for the LM Fuser Framework that integrates captions generated by many MLLMs wusing their pretrained weightd and then integrates the information training a smaller LM only on the outputs of these MLLMs.
- Code for the LM-Fuser2 which is an alternative version of LM-Fuser in which we don't use the outputs but the most similar candidate caption in each beam step.

## Experiments

Experiments were conducted on ImageCLEF 2024 dataset, including [specify dataset names here, e.g., Chest X-ray, MIMIC-CXR]. The results of these experiments demonstrate the effectiveness of MLLMs in generating accurate and meaningful captions for medical images.


## Models Performance Comparison

| Model                | BertScore | ROUGE-L | BLEURT | BLEU  |
|----------------------|-----------|---------|--------|-------|
| InstructBLIP         | 61.64     | 19.31   | 27.85  | 10.0  |
| LLaVA 1.5            | 62.08     | 22.53   | 30.99  | 20.8  |
| Llama 3.1 Synthesizer| 63.55     | 23.29   | 31.44  | 21.7  |
| LM-Fuser2            | 65.10     | 24.24   | 32.25  | 20.2  |
| LM-Fuser             | 65.27     | 24.40   | 31.99  | 21.4  |


## Citation

If you use this repository or parts of this work in your own research, please cite the following:

```
@mastersthesis{msamprovalaki-thesis,
 author = {Marina Samprovalaki},
 school = {Athens University of Economics and Business},
 title = {Exploring Multimodal Large Language Models for Medical Image Captioning},
 url = {tba},
 year = {2024}
}
```
