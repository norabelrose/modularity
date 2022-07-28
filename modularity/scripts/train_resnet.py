from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models.regnet import AnyStage
from torch.utils.data import DataLoader
from ..crossover import Crossover
from ..resnet_model import ResNetModel
import pytorch_lightning as pl
import torchvision as tv


def main():
    parser = ArgumentParser()
    parser.add_argument("name", help="Name of the experiment")
    parser.add_argument("--acc-grad-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
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
        "--lr", type=float, default=0.05, help="Initial learning rate for training"
    )
    parser.add_argument(
        "--num-alleles", type=int, default=2, help="Number of alleles to recombine"
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

    train_dl = DataLoader(
        train, batch_size=args.batch_size, num_workers=8, shuffle=True
    )
    val_dl = DataLoader(val, batch_size=args.batch_size, num_workers=8)

    idx = 0

    def filter_fn(model):
        nonlocal idx
        if isinstance(model, AnyStage):
            idx += 1
            return idx % 3 == 2

        return False

    model = ResNetModel(10, sd_prob=args.sd_prob)
    if args.num_alleles:
        model = Crossover.wrap_recursive(
            model,
            filter_fn,
        )
    trainer = pl.Trainer(
        gpus=args.devices,
        accumulate_grad_batches=args.acc_grad_batches,
        callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")],
        logger=TensorBoardLogger(args.log_dir, version=args.name),
        max_epochs=100,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
