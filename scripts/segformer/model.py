import pytorch_lightning as pl
import torch
from transformers import SegformerForSemanticSegmentation
import segformer.config as config
from torch import nn
import evaluate
import time
import json 
import numpy as np

class SegformerFinetuner(pl.LightningModule):

    def __init__(self, id2label, lr ):
        super(SegformerFinetuner, self).__init__()
        self.lr=lr
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
        self.log("trainLoss", loss, sync_dist=True,  batch_size=config.BATCH_SIZE)
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)   
        loss, logits = outputs[0], outputs[1]
        self.log("valLoss", loss, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True,logger=True, prog_bar=True)
        # Retrieve current learning rate
        # Assuming only one group in optimizer
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)

        
        return loss
    
        
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
        
        # Calculate FN and FP
        false_negatives = np.sum((predicted.detach().cpu().numpy() == 0) & (masks.detach().cpu().numpy() == 1))
        false_positives = np.sum((predicted.detach().cpu().numpy() == 1) & (masks.detach().cpu().numpy() == 0))
        
        # Total number of instances
        total_instances = np.prod(predicted.shape)
        
        # Calculate percentages
        percentage_fn = (false_negatives / total_instances) 
        percentage_fp = (false_positives / total_instances) 
    
        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'testLoss': loss, 
            "mean_iou": metrics["mean_iou"], 
            "mean_accuracy": metrics["mean_accuracy"],
            "False Negative": percentage_fn,
            "False Positive": percentage_fp,
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k,v in metrics.items():
            self.log(k,v,sync_dist=True)
        return(metrics)
        
    def configure_optimizers(self):
        # AdamW optimizer with specified learning rate
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)

        # ReduceLROnPlateau scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.FACTOR, patience=config.PATIENCE),
            'reduce_on_plateau': True,  # Necessary for ReduceLROnPlateau
            'monitor': 'valLoss'  # Metric to monitor for reducing learning rate
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
