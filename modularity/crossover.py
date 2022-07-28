from copy import deepcopy
from typing import cast, Callable, Literal, Type, Union

import torch as th


class Crossover(th.nn.Module):
    def __init__(
        self,
        template: th.nn.Module,
        num_alleles: int = 2,
        *,
        eval_allele: int = 0,
        mode: Literal["batch", "row"] = "batch",
    ):
        super().__init__()

        if eval_allele < 0 or eval_allele >= num_alleles:
            raise ValueError(f"Index of eval allele must be in [0, {num_alleles - 1}].")

        self.alleles = th.nn.ModuleList(
            [deepcopy(template) for _ in range(num_alleles)]
        )
        self.eval_allele = eval_allele
        self.mode = mode
        self.reset_parameters()

    def forward(self, *args, **kwargs):
        # During training, randomly select an allele to use
        if self.training:
            # Special case for the "row" mode
            if self.mode == "row":
                batch_size = args[0].shape[0]
                assert batch_size % self.num_alleles == 0

                indices = th.randint(
                    0, len(self.alleles), (batch_size,), device=args[0].device
                )

            idx = th.randint(0, len(self.alleles), ())
            active = self.alleles[idx]

        # During evaluation, deterministically select the eval allele
        else:
            active = self.alleles[self.eval_allele]

        return active(*args, **kwargs)

    def reset_parameters(self):
        """Reset the parameters of all alleles."""

        # Make sure no parameter is left behind
        dirty_params = {v: k for k, v in self.named_parameters()}

        def maybe_reset(module: th.nn.Module):
            nonlocal dirty_params

            # There's no point in nesting Crossover wrappers since they
            # would be equivalent to a single wrapper with a larger number
            # of alleles. Prevent the user from making this mistake.
            if isinstance(module, Crossover):
                raise ValueError("Nesting of Crossover wrappers is not supported.")

            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()

                # Check these parameters off our list
                for param in module.parameters(recurse=False):
                    del dirty_params[param]

        self.alleles.apply(maybe_reset)
        if dirty_params:
            raise ValueError(
                f"Some parameters could not be reset. Make sure to implement "
                f"`reset_parameters` when necessary on custom nn.Module "
                f"subclasses. Affected params: {sorted(dirty_params.values())}"
            )

    def __repr__(self):
        # Flipping the parameter order to make it more readable
        template = self.alleles[self.eval_allele]
        return f"Crossover(num_alleles={len(self.alleles)}, template={template})"

    @classmethod
    def wrap_recursive(
        cls,
        root: th.nn.Module,
        filter_fn: Union[Callable[[th.nn.Module], bool], Type[th.nn.Module]],
        *,
        num_alleles: int = 2,
    ) -> th.nn.Module:
        """Wrap all the descendants of a module that match some criterion in
        Crossover layers.
        """
        if isinstance(filter_fn, type):
            filter_fn = lambda m: isinstance(m, cast(type, filter_fn))

        for name, module in root.named_children():
            patched = cls.wrap_recursive(module, filter_fn, num_alleles=num_alleles)
            setattr(root, name, patched.train(module.training))

        return cls(root, num_alleles) if filter_fn(root) else root


# class ShuffledTensor(th.Tensor):
#     """A tensor that has been permuted across the batch dimension. It keeps
#     track of this permutation in the `indices` attribute."""
#
#     def shuffle(self):
#         """Shuffle the tensor in-place."""
#         perm = th.randperm(self.shape[0])
#         self[:] = self[perm]
#
#     def __repr__(self):
#         return f"ShuffledTensor({self.shape})"
