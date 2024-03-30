# -Finetuning SegFormer on custom Dataset

## Introduction
SegFormer is a simple and Efficient Design for Semantic Segmentation with Transformer which unifies Transformers with lightweight multilayer perception (MLP) decoders. 
MMSegmentation v0.13.0 is used as the codebase.

### [Project page](https://github.com/NVlabs/SegFormer) | [Paper](https://arxiv.org/abs/2105.15203) | [Demo (Youtube)](https://www.youtube.com/watch?v=J0MoRQzZe8U) | [Demo (Bilibili)](https://www.bilibili.com/video/BV1MV41147Ko/) | [Intro Video](https://www.youtube.com/watch?v=nBjXyoltCHU)
## Purpose
The purpose of this document is to build a process of finetuning Segformer for custom dataset on semantic segmentation. The code is done using Pytorch Lightning and the model can be imported from hugging face.

## Installation
For install and data preparation, please refer to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```pip install timm==0.3.2```

An example (works for me): ```CUDA 10.1``` and  ```pytorch 1.7.1``` 

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd SegFormer && pip install -e . --user
```
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
