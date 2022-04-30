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

@hydra.main(config_path="../config", config_name="default")
def main(config):
    ds = torchvision.datasets.CIFAR100("../data/cifar", download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    data_collator = DiffusionCollator()
    dl = DataLoader(ds, batch_size=config.training.batch_size, collate_fn=data_collator)

    model = DiffusionModule(
        lr=config.training.lr,
    )

    trainer = pl.Trainer(
        max_steps=config.training.steps
    )
    trainer.fit(model, dl)

    print("done")

    # epoch = 0
    # with tqdm(total=config.training.steps) as pbar:
    #     epoch += 1
    #     for batch in dl:
    #         loss = model.get_loss(batch)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         pbar.set_postfix(epoch=epoch, loss=loss.item())

if __name__ == '__main__':
    main()