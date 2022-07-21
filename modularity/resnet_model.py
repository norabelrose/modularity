from types import MethodType
from typing import Literal
import pytorch_lightning as pl
import torch as th
import torch.nn.functional as F
import torchvision as tv


class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes: int, sd_prob: float = 0.0):
        """
        Args:
            sd_prob: Drop probability for each layer when using stochastic
                depth. If 0, no stochastic depth is used.
        """
        super().__init__()
        self.resnet = tv.models.regnet_x_1_6gf(num_classes=num_classes)

        # Manually patch the network to include stochastic depth
        def patched_forward(self, x: th.Tensor) -> th.Tensor:
            diff = tv.ops.stochastic_depth(self.f(x), sd_prob, "row", self.training)
            if self.proj is not None:
                x = self.proj(x) + diff
            else:
                x = x + diff

            return self.activation(x)

        for mod in self.resnet.modules():
            if isinstance(mod, tv.models.regnet.ResBottleneckBlock):
                mod.forward = MethodType(patched_forward, mod)

    def configure_optimizers(self):
        sgd = th.optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-5)
        return [sgd], [th.optim.lr_scheduler.CosineAnnealingLR(sgd, 100)]

    def shared_step(
        self, batch: tuple[th.Tensor, th.Tensor], stage: Literal["train", "val", "test"]
    ) -> th.Tensor:
        images, labels = batch
        logits = self.resnet(images)
        loss = F.cross_entropy(logits, labels)

        hits = logits.argmax(-1) == labels
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_acc", hits.float().mean())
        return loss

    def training_step(
        self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int
    ) -> th.Tensor:
        return self.shared_step(batch, "train")

    def validation_step(
        self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int
    ) -> th.Tensor:
        return self.shared_step(batch, "val")

    def test_step(
        self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int
    ) -> th.Tensor:
        return self.shared_step(batch, "test")
