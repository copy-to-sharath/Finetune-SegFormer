# %%

from torch.utils.data import Dataset
import os
from PIL import Image
from pycocotools.coco import COCO

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train
        sub_path = "train" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir,  sub_path)
        self.mask_dir = os.path.join(self.root_dir, "masks",sub_path)
        self.coc_ann_dir = os.path.join(self.root_dir, "annotations",f"instances_{sub_path}.json")        
        self.coco =COCO(self.coc_ann_dir)
        self.id2label ={cat[1]['id']:cat[1]['name'] for cat in self.coco.cats.items()}
        self.label2id ={cat[1]['name']:cat[1]['id'] for cat in self.coco.cats.items()}
        self.cats = self.coco.cats
        self.num_classes = len(self.id2label)

        # # read images
        # image_file_names = []
        # for root, dirs, files in os.walk(self.img_dir):
        #   image_file_names.extend(files)
        # self.images = sorted(image_file_names)

        # # read annotations
        # annotation_file_names = []
        # for root, dirs, files in os.walk(self.ann_dir):
        #   annotation_file_names.extend(files)
        # self.annotations = sorted(annotation_file_names)

        # image_file_names = [f for f in os.listdir(self.img_dir) if '.jpg' in f]
        # mask_file_names = [f for f in os.listdir(self.mask_dir) if '.png' in f]


        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        mask_file_names = []
        for root, dirs, files in os.walk(self.mask_dir):
          mask_file_names.extend(files)
        self.masks = sorted(mask_file_names)

        # assert len(self.images) == len(self.masks), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

      image = Image.open(os.path.join(self.img_dir, self.images[idx]))
      segmentation_map = Image.open(os.path.join(self.mask_dir, self.masks[idx]))

      # randomly crop + pad both image and segmentation map to same size
      encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

      for k,v in encoded_inputs.items():
        encoded_inputs[k].squeeze_() # remove batch dimension

      return encoded_inputs
# %%

"""Let's initialize the training + validation datasets. Important: we initialize the image processor with `reduce_labels=True`, as the classes in ADE20k go from 0 to 150, with 0 meaning "background". However, we want the labels to go from 0 to 149, and only train the model to recognize the 150 classes (which don't include "background"). Hence, we'll reduce all labels by 1 and replace 0 by 255, which is the `ignore_index` of SegFormer's loss function."""


from transformers import SegformerImageProcessor

root_dir = '/home/sharath/data/YOLO-10000-BOX.v6i.coco'
image_processor = SegformerImageProcessor(reduce_labels=True)

train_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=True)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=False)

id2label = train_dataset.id2label
label2id= train_dataset.label2id
num_classes = train_dataset.num_classes

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))


# %%
"""Let's verify a random example:"""
# encoded_inputs = train_dataset[0]
# encoded_inputs["pixel_values"].shape
# encoded_inputs["labels"].shape
# encoded_inputs["labels"]
# encoded_inputs["labels"].squeeze().unique()

# %%

"""Next, we define corresponding dataloaders."""

from IPython.display import display
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=12,shuffle=False)

# batch = next(iter(train_dataloader))

# for k,v in batch.items():
#   print(k, v.shape)

# batch["labels"].shape

# mask = (batch["labels"] != 255)
# print(mask)

# batch["labels"][mask]
# print(mask)

# %%

"""## Define the model
Here we load the model, and equip the encoder with weights pre-trained on ImageNet-1k (we take the smallest variant, `nvidia/mit-b0` here, but you can take a bigger one like `nvidia/mit-b5` from the [hub](https://huggingface.co/models?other=segformer)). We also set the `id2label` and `label2id` mappings, which will be useful when performing inference.
"""

from transformers import SegformerForSemanticSegmentation

# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=num_classes,
                                                         id2label=id2label,
                                                         label2id=label2id,
)

"""## Fine-tune the model

Here we fine-tune the model in native PyTorch, using the AdamW optimizer. We use the same learning rate as the one reported in the [paper](https://arxiv.org/abs/2105.15203).

It's also very useful to track metrics during training. For semantic segmentation, typical metrics include the mean intersection-over-union (mIoU) and pixel-wise accuracy. These are available in the Datasets library. We can load it as follows:
"""

import evaluate

metric = evaluate.load("mean_iou")

image_processor.do_reduce_labels

import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm

os.makedirs("checkpoints", exist_ok=True)

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(500):  # loop over the dataset multiple times
  print("Epoch:", epoch)
  for idx, batch in enumerate(tqdm(train_dataloader)):
    # get the inputs;
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(pixel_values=pixel_values, labels=labels)
    loss, logits = outputs.loss, outputs.logits

    loss.backward()
    optimizer.step()

    # evaluate
    with torch.no_grad():
      upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
      predicted = upsampled_logits.argmax(dim=1)

      # note that the metric expects predictions + labels as numpy arrays
      metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

      # let's print loss and metrics every 100 batches
      if idx % 100 == 0:
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
                predictions=predicted.cpu(),
                references=labels.cpu(),
                num_labels=len(id2label),
                ignore_index=255,
                reduce_labels=False, # we've already reduced the labels ourselves
            )

        print("Loss:", loss.item())
        print("Mean_iou:", metrics["mean_iou"])
        print("Mean accuracy:", metrics["mean_accuracy"])
  model_path =f"checkpoints/epoch_{epoch}.pth"
  torch.save(model, model_path)

# %%
