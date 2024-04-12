import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from segformer.model import SegformerFinetuner
from segformer.dataset import SegmentationDataModule
import config
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os

def dataset_predictions(dataloader):
    pred_set=[]
    label_set=[]
    for batch in tqdm((dataloader), desc="Doing predictions"):
        images, labels = batch['pixel_values'], batch['labels']
        outputs = model(images, labels)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            #size of original image is 640x640
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted_mask = upsampled_logits.argmax(dim=1).numpy()
        labels = labels.numpy()
        pred_set.append(predicted_mask)
        label_set.append(labels)        
    return pred_set, label_set

def savePredictions(pred_set, label_set, save_path):
    for i in tqdm(range(len(pred_set)), desc="Saving predictions"):
        file_name = f"result_{i}"
        n_plots = len(pred_set[i])  # Assuming this gives the number of items per batch
        
        # Dynamically adjust subplot layout based on batch size
        f, axarr = plt.subplots(n_plots, 2)  # Two columns for predictions and ground truth
        f.set_figheight(15)
        f.set_figwidth(15)
        
        # Set titles only if there's more than one subplot; otherwise, adjust indexing for axarr
        if n_plots > 1:
            axarr[0, 0].set_title("Predictions", {'fontsize': 30})
            axarr[0, 1].set_title("Ground Truth", {'fontsize': 30})
        else:
            axarr[0].set_title("Predictions", {'fontsize': 30})
            axarr[1].set_title("Ground Truth", {'fontsize': 30})
        
        for j in range(n_plots):
            # Adjust for when there's only a single plot
            if n_plots > 1:
                axarr[j, 0].imshow(pred_set[i][j, :, :])
                axarr[j, 1].imshow(label_set[i][j, :, :])
            else:
                axarr[0].imshow(pred_set[i][j, :, :])
                axarr[1].imshow(label_set[i][j, :, :])

        # Construct the full path where the image will be saved
        file_path = os.path.join(save_path, f"{file_name}.png")

        # Save the figure
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(f)  # Close the figure to free memory
        
    print("Predictions saved")

if __name__=="__main__":
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--model_path',
    type=str,
    default='',
    help="Enter the path of your model.ckpt file"
    )
    parser.add_argument(
    '--save_path',
    type=str,
    default='',
    help="enter the path to save your images"
    )

    args = parser.parse_args()
    model_path = args.model_path
    save_path = args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    model = SegformerFinetuner.load_from_checkpoint(model_path,id2label=config.ID2LABEL, lr=config.LEARNING_RATE)
    
    model.eval()
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    pred_set, label_set= dataset_predictions(test_dataloader)
    savePredictions(pred_set, label_set, save_path)
        
    
