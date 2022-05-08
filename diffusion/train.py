import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import hydra
import pytorch_lightning as pl

from model import DiffusionModule

class DiffusionCollator:
    def __call__(self, examples):
        x, _ = list(zip(*examples))
        x = torch.stack(x)
        return x

@hydra.main(config_path="../config", config_name="diffusion")
def main(config):
    train_ds = torchvision.datasets.CIFAR100("../data/cifar", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    val_ds = torchvision.datasets.CIFAR100("../data/cifar", download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    data_collator = DiffusionCollator()
    train_dl = DataLoader(train_ds, pin_memory=True, num_workers=config.training.num_workers, batch_size=config.training.batch_size, collate_fn=data_collator, shuffle=True)
    val_dl = DataLoader(val_ds, pin_memory=True, num_workers=config.training.num_workers, batch_size=config.training.batch_size, collate_fn=data_collator)

    model = DiffusionModule(
        lr=config.training.lr,
    )

    logger = pl.loggers.WandbLogger(project="music-gen", config=config)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        max_epochs=config.training.epochs,
        logger=logger,
        log_every_n_steps=100,
        val_check_interval=1000,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    main()