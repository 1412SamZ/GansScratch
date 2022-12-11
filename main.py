from datasets.ganDataset import MNISTDataModule
from models.GAN import GAN

import pytorch_lightning as pl 
import torch 


dm = MNISTDataModule()
model = GAN()

# model.plot_imgs(False)


trainer = pl.Trainer(max_epochs=1, gpus=1,enable_checkpointing=False)
trainer.fit(model, dm)
trainer.save_checkpoint("./saved_models/GANFinal.ckpt")

