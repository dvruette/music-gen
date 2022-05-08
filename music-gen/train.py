import glob
import os
import warnings

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import hydra
from hydra.utils import to_absolute_path

from data import AudioDataset, AudioCollator
from model import AudioModule

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

def get_train_test_files(root_dir, train_split=0.8):
    files = glob.glob(os.path.join(root_dir, "**", "*.mp3"), recursive=True)
    n_train = int(train_split * len(files))
    return files[:n_train], files[n_train:]

@hydra.main(config_path="../config", config_name="music-gen")
def main(config):
    root_dir = to_absolute_path(config.data.root_dir)
    train_files, val_files = get_train_test_files(root_dir)

    train_ds = AudioDataset(train_files, sr=config.data.sample_rate)
    val_ds = AudioDataset(val_files, sr=config.data.sample_rate)

    chunk_size = (config.model.n_fft//2 + 1) * config.model.seq_len
    data_collator = AudioCollator(chunk_size=chunk_size)
    train_dl = DataLoader(train_ds, pin_memory=True, num_workers=config.training.num_workers, batch_size=config.training.batch_size, collate_fn=data_collator, shuffle=True)
    val_dl = DataLoader(val_ds, pin_memory=True, num_workers=config.training.num_workers, batch_size=config.training.batch_size, collate_fn=data_collator)

    model = AudioModule(
        n_fft=config.model.n_fft,
        seq_len=config.model.seq_len,
        lr=config.training.lr,
    )

    logger = pl.loggers.WandbLogger(project="music-gen", config=config)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=config.training.epochs,
        logger=logger,
        log_every_n_steps=100,
        val_check_interval=0.2,
        callbacks=[lr_monitor]
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    main()