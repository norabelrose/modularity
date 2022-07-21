from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from ..resnet_model import ResNetModel
import pytorch_lightning as pl
import torchvision as tv


def main():
    parser = ArgumentParser()
    parser.add_argument("name", help="Name of the experiment")
    parser.add_argument(
        "--devices", type=int, default=[0], nargs="+", help="Device IDs of GPUs to use"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="modularity-logs",
        help="Directory to save checkpoints in",
    )
    parser.add_argument(
        "--sd-prob", type=float, default=0.0, help="Stochastic depth drop probability"
    )
    args = parser.parse_args()

    train_transform = tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0, 0, 0], [1, 1, 1]),
        ]
    )
    val_transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0, 0, 0], [1, 1, 1]),
        ]
    )

    train = tv.datasets.CIFAR10(
        "cifar10_train", train=True, download=True, transform=train_transform
    )
    val = tv.datasets.CIFAR10(
        "cifar10_val", train=False, download=True, transform=val_transform
    )

    train_dl = DataLoader(train, batch_size=128, num_workers=8, shuffle=True)
    val_dl = DataLoader(val, batch_size=128, num_workers=8)

    model = ResNetModel(10, sd_prob=args.sd_prob)
    trainer = pl.Trainer(
        gpus=args.devices,
        callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")],
        logger=TensorBoardLogger(args.log_dir, version=args.name),
        max_epochs=100,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
