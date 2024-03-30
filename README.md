# Finetuning SegFormer on custom Dataset

## Introduction
SegFormer is a simple and Efficient Design for Semantic Segmentation with Transformer which unifies Transformers with lightweight multilayer perception (MLP) decoders. 
MMSegmentation v0.13.0 is used as the codebase.

### [SegFormer Project page](https://github.com/NVlabs/SegFormer) | [ SegFormer Paper](https://arxiv.org/abs/2105.15203) | [Demo (Youtube)](https://www.youtube.com/watch?v=J0MoRQzZe8U) | [Demo (Bilibili)](https://www.bilibili.com/video/BV1MV41147Ko/) | [Intro Video](https://www.youtube.com/watch?v=nBjXyoltCHU)
### [SegFormer Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/segformer)

## Purpose
The purpose of this document is to build a process of finetuning Segformer for custom dataset on semantic segmentation. The code is done using Pytorch Lightning and the model can be imported from hugging face.

1. Create a virtual environment: `conda create -n segformer python=3.10 -y` and `conda activate segformer `
2. Install [Pytorch CUDA 11.8](https://pytorch.org/): ` pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
3. Download code: `git clone https://github.com/bowang-lab/U-Mamba`
5. `cd U-Mamba/umamba` and run `pip install -e .`

## Evaluation
The Evaluation is done using Mean-IOU from the evaluate package.

## Visualisation

## Citation
```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
