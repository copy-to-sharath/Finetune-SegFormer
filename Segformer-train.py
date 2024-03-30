import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader 
import torch
torch.manual_seed(1)
from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import torchmetrics
from torchmetrics import Metric
from datasets import load_metric
from torch import nn
import random
torch.set_float32_matmul_precision("medium")
import evaluate
import time
import json 

class ImageSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.filenames[idx] + '.jpg')
        mask_path = os.path.join(self.masks_dir, self.filenames[idx] + '.png') 
        image = Image.open(img_path).convert("RGB")
        np_image=np.array(image)
        # convert to C, H, W
        np_image = np_image.transpose(2,0,1)
        mask = Image.open(mask_path) 
        np_mask=np.array(mask)
        np_mask[np_mask==255]=1
      
        return np_image, np_mask
    
def collate_fn(batch):
    inputs = list(zip(*batch))
    images=inputs[0]
    segmentation_maps=inputs[1]
    batch = processor(
        images,
        segmentation_maps=segmentation_maps,
        size=(640,640),
        return_tensors="pt",
    )
    return batch

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, num_workers, processor=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processor = processor
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'images', 'train'),
                                                          masks_dir=os.path.join(self.dataset_dir, 'labels', 'train'),
                                                          transform=None) # Add your transforms here
            self.val_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'images', 'val'),
                                                        masks_dir=os.path.join(self.dataset_dir, 'labels', 'val'),
                                                        transform=None) # Add your transforms here
        if stage == 'test' or stage is None:
            self.test_dataset = ImageSegmentationDataset(images_dir=os.path.join(self.dataset_dir, 'images', 'test'),
                                                         masks_dir=os.path.join(self.dataset_dir, 'labels', 'test'),
                                                         transform=None) # Add your transforms here
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
    
class SegformerFinetuner(pl.LightningModule):

    def __init__(self, id2label):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        evaluate.load
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")

    def forward(self, pixel_values, labels):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return(outputs)
    '''
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        batch['pixel_values'] = batch['pixel_values'].to(device)
        batch['labels'] = batch['labels'].to(device)
        return batch
    '''
    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        total_time = time.time() - self.start_time
        metrics = {'final_epoch': self.current_epoch, 'training_time': total_time}
        with open('segformer_hyperparameters.json', 'w') as f:
            json.dump(metrics, f)

    def training_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.train_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
            num_labels=self.num_classes,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
    
        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'loss': loss, 
            "mean_iou": metrics["mean_iou"], 
            "mean_accuracy": metrics["mean_accuracy"],
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k,v in metrics.items():
            self.log(k,v,sync_dist=True, on_epoch=True, logger=True)
        return(metrics)

    def validation_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)   
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks[0].shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.val_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
            num_labels=self.num_classes,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
    
        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'loss': loss, 
            "mean_iou": metrics["mean_iou"], 
            "mean_accuracy": metrics["mean_accuracy"],
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k,v in metrics.items():
            self.log(k,v,sync_dist=True)
        return(metrics)
        
    def test_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.test_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
            num_labels=self.num_classes,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
    
        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'loss': loss, 
            "mean_iou": metrics["mean_iou"], 
            "mean_accuracy": metrics["mean_accuracy"],
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k,v in metrics.items():
            self.log(k,v,sync_dist=True)
        return(metrics)
        
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=0.001)
        


batch_size=1
num_workers=4
newid2label={
    0: 'Background',
    1: 'Metal Lines'
}
early_stop_callback = EarlyStopping(
    monitor="loss",
    min_delta=0.00,
    patience=4,
    verbose=True,
    mode="min",
)

checkpoint_callback = ModelCheckpoint(save_top_k=1, 
                                      monitor="loss", 
                                      every_n_epochs=1,  # Save the model at every epoch 
                                      save_on_train_epoch_end=True  # Ensure saving happens at the end of a training epoch
                                     )
processor = SegformerImageProcessor()
data_module = SegmentationDataModule(dataset_dir='data_augmentation_dataset', batch_size=batch_size, num_workers=num_workers, processor=processor)
#path="mask2former_checkpoint_14Mar.ckpt"
#model = SegFormerFinetuner.load_from_checkpoint(path,id2label=newid2label)
model=SegformerFinetuner(newid2label)
logger_csv = pl.loggers.CSVLogger("outputs", name="lightning_logs_csv")

trainer = pl.Trainer(
    logger=logger_csv,
    precision="16-mixed",
    accelerator='cuda',
    devices=[1,2,3],
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=100
)
print("Training starts!!")
trainer.fit(model,data_module)
print("saving model!")
trainer.save_checkpoint("segformer_checkpoint_hyperparameters.ckpt")