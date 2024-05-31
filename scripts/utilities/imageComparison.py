import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segformer import  (SegformerFinetuner, 
                        SegmentationDataModule, 
                        DATASET_DIR, 
                        BATCH_SIZE, 
                        NUM_WORKERS, 
                        ID2LABEL, 
                        LEARNING_RATE)
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from colorPalette import color_palette, apply_palette
 

def dataset_predictions(dataloader):
    pred_set=[]
    image_set=[]
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
        images = images.numpy()
        pred_set.append(predicted_mask)
        image_set.append(images)  
        label_set.append(labels)      
    return pred_set, image_set, label_set

def savePredictions(pred_set, image_set, label_set, save_path):
    palette = color_palette()  # Ensure this function is defined or imported appropriately
    index = 0
    
    for batch_index in tqdm(range(len(pred_set)), desc="Saving predictions"):
        for image_index in range(len(pred_set[batch_index])):
            prediction = pred_set[batch_index][image_index]
            image = image_set[batch_index][image_index]
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
            label = label_set[batch_index][image_index]
            image = np.transpose(image, (1, 2, 0))  # Convert image from (C, H, W) to (H, W, C)

            colored_prediction = apply_palette(prediction, palette)
            color_seg = apply_palette(prediction, palette)

            # Create an overlay image by blending the original image with the colored segmentation mask
            overlay_image = (image * 0.5 + color_seg * 0.5).astype(np.uint8)

            file_name = f"result_{index}"
            f, axarr = plt.subplots(1, 3, figsize=(22, 7.5))  # One row, three columns
            axarr[0].imshow(colored_prediction)
            axarr[0].set_title("Predictions", fontsize=20)
            axarr[0].axis('off')

            axarr[1].imshow(overlay_image)
            axarr[1].set_title("Overlay Image", fontsize=20)
            axarr[1].axis('off')

            axarr[2].imshow(image)
            axarr[2].set_title("Original Image", fontsize=20)
            axarr[2].axis('off')

            plt.savefig(os.path.join(save_path, f"{file_name}.png"), bbox_inches='tight')
            plt.close(f)
            index += 1

    print("Predictions saved")


if __name__=="__main__":
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
    model_path = os.path.join( '..', args.model_path)
    save_path = os.path.join( '..', args.save_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model = SegformerFinetuner.load_from_checkpoint(model_path,id2label=ID2LABEL, lr=LEARNING_RATE)
    model.eval()
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    pred_set, image_set, label_set = dataset_predictions(test_dataloader)
    savePredictions(pred_set, image_set,label_set, save_path)
        
    
