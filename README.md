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
3. Download code: `git clone https://github.com/sleepreap/Finetuning-SegFormer.git`
4. `cd Finetuning-SegFormer` and run `pip install -e .`

## Dataset
Use createDataset.py to create the folders.
Refer to the README file in the folder "Data" on where to upload the images and labels.

## Training
1. 'cd scripts' 
2. set up the configs required in config.py
3. run the train.py file

A CSVlogger and the trained model file will be saved after the training has been completed. The model file would be saved as "segformer_checkpoint_hyperparameters.ckpt" in the same directory. An "output" folder will be created to store the contents of the CSVlogger.

## Testing
The testing is done using Mean-IOU, as well as pixel accuracy from the evaluate package. It will provide individual accuracy and IOU scores for each class label specified, as well as the mean accuracy and IOU scores of all the class labels. To run the test file, the model path of the trained model must be provided as an argument.

1. 'cd scripts' 
2. run the test.py file using this command: python test.py --model_path MODEL_PATH
   
```bash
e.g python test.py --model_path segformer_checkpoint_hyperparameters.ckpt
```
![image](https://github.com/sleepreap/Finetuning-SegFormer/assets/98008874/9de7ce23-c06e-4652-8a48-1ff84986ef04)

## Utilities
This folder contains the following scripts:
1. inference.py
2. saveComparison.py
3. predictionOverlay.py
4. saveComparisonWithOverlay.py
   
### All the scripts already reference the parent folder for the command line arguments. The images and labels used are from the test dataset respectively.

Inference.py would save all the predictions by the model on the test dataset in the specified save path folder.

![image](https://github.com/sleepreap/Finetuning-SegFormer/assets/98008874/f53bef82-f5e1-42c6-a3a1-a422980dac0a)

```bash
1. 'cd scripts/utilities'
2. run the inference.py file using this command: python inference.py --model_path MODEL_PATH --save_path SAVE_PATH
```

saveComparison.py would save a plot of the prediction and the ground truth side by side in the specified save path folder. The number of subplots is based on batch_size defined in the config file. 

![image](https://github.com/sleepreap/Finetuning-SegFormer/assets/98008874/47a694d1-d757-413d-8608-08442e37817b)

```bash
1. 'cd scripts/utilities'
2. run the saveComparison.py file using this command: python saveComparison.py --model_path MODEL_PATH --save_path SAVE_PATH
```

predictionOverlay.py would save the overlay that shows the TP+TN+FP+FN of the predictions done by the model for all the images in the specified save path folder. Black means TN (background), Green means TP (metal-line), Red means FN (metal-line as background), Blue means FP (background as metal-line).

![image](https://github.com/sleepreap/Finetuning-SegFormer/assets/98008874/1922882e-f04a-42c4-af7c-25f08b33873b)


```bash
1. 'cd scripts/utilities'
2. run the predictionOverlay.py file using this command: python predictionOverlay.py --model_path MODEL_PATH --save_path SAVE_PATH
```

saveComparisonWithOverlay.py would save a plot of the overlay and the ground truth side by side in the specified save path folder. The number of subplots is based on batch_size defined in the config file.

```bash
1. 'cd scripts/utilities'
2. run the saveComparisonWithOverlay.py file using this command: python saveComparisonWithOverlay.py --model_path MODEL_PATH --save_path SAVE_PATH
```

## Citation
```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
