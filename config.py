from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

#Training hyperparmeters
LEARNING_RATE=0.001
EPOCHS=100
PRECISION="16-mixed"
DEVICES=[1,2,3]
EARLY_STOPPING_CALLBACK = EarlyStopping(
    monitor="loss",
    min_delta=0.00,
    patience=4,
    verbose=True,
    mode="min",
)
CHECKPOINT_CALLBACK = ModelCheckpoint(save_top_k=1, 
                                      monitor="loss", 
                                      every_n_epochs=1,  # Save the model at every epoch 
                                      save_on_train_epoch_end=True  # Ensure saving happens at the end of a training epoch
                                     )
LOGGER = CSVLogger("outputs", name="lightning_logs_csv")

#Dataset
DATASET_DIR='data/dataset'
NUM_WORKERS=4
BATCH_SIZE=1
ID2LABEL={
    0: 'Background',
    1: 'Metal Lines'
}
