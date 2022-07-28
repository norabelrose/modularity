from collections import defaultdict
from functools import partial
from typing import Callable, Type, Union
import torch as th


ModuleFilter = Union[Callable[[th.nn.Module], bool], Type[th.nn.Module]]


def is_torch_activation(module: th.nn.Module) -> bool:
    """Check if a module is a built-in PyTorch activation function. This
    should be robust to the addition of new activation functions."""
    mod_name = type(module).__module__
    return mod_name == "torch.nn.modules.activation"


class ActivationListener:
    """Records intermediate activations of a model for later inspection.

    Args:
        model: The model to record activations for.
        filter_fn: A function that takes a module and returns True if its
            input or output should be recorded.

        include_pre: Whether to record the pre-activations / inputs.
        include_post: Whether to record the post-activations / outputs.
    """

    def __init__(
        self,
        model: th.nn.Module,
        filter_fn: ModuleFilter = is_torch_activation,
        *,
        include_pre: bool = False,
        include_post: bool = True,
    ):
        # Passing a class is shorthand for a lambda isinstance check
        if isinstance(filter_fn, type):
            self.filter_fn = lambda m: isinstance(m, filter_fn)
        else:
            self.filter_fn = filter_fn

        if not include_pre and not include_post:
            raise ValueError(
                "At least one of include_pre or include_post must be True."
            )

        self.activations = defaultdict(list)
        self.preactivations = defaultdict(list)
        self.hooks = {}
        self.include_post = include_post
        self.include_pre = include_pre
        self.model = model
        self.attach()

    def attach(self):
        """Attach forward hooks to the model."""

        def hook(m, i, o, param_name):
            if self.include_post:
                self.activations[param_name].append(o.detach())
            if self.include_pre:
                self.preactivations[param_name].append(i.detach())

        for name, module in self.model.named_modules():
            if not self.filter_fn(module):
                continue

            self.hooks[name] = module.register_forward_hook(
                partial(hook, param_name=name)
            )

    def detach(self):
        """Remove all forward hooks from the model."""
        for hook in self.hooks.values():
            hook.remove()

        self.hooks.clear()

    def num_activations(self) -> int:
        """Return the number of activations recorded."""
        return sum(
            sum(act.numel() for act in acts) for acts in self.activations.values()
        )

    def offload(self):
        """Offload the cached activations to the CPU."""
        for name, acts in self.activations.items():
            self.activations[name] = [act.cpu() for act in acts]

        for name, acts in self.preactivations.items():
            self.preactivations[name] = [act.cpu() for act in acts]

    def reset(self):
        """Clear the activations cache."""
        self.activations.clear()
        self.preactivations.clear()
