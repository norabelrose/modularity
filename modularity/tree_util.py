import torch as th
from numbers import Number
from typing import cast, Any, Callable, Iterable, Sequence, TypeVar, Union


def pytree_batch_size(tree: Any) -> int:
    """Finds the first tensor in a pytree and returns its batch size."""
    for elem in pytree_flatten(tree):
        if isinstance(elem, th.Tensor) and elem.ndim > 0:
            return elem.shape[0]

    raise ValueError("No tensor found in pytree.")


def pytree_flatten(tree: Any) -> Iterable[Union[Number, th.Tensor]]:
    """Recursively iterate over all tensors in a pytree, in topological order."""
    # Stopping condition
    if isinstance(tree, (Number, th.Tensor)):
        yield tree

    # Recursive case
    elif isinstance(tree, dict):
        for elem in tree.values():
            yield from pytree_flatten(elem)

    elif isinstance(tree, Sequence):
        for elem in tree:
            yield from pytree_flatten(elem)


LeafInput = TypeVar("LeafInput", bool, complex, float, int, th.Tensor)


def pytree_map(func: Callable[[LeafInput], Any], tree: Any, strict: bool = True):
    """
    Recursively apply a function to all tensors in a pytree, returning the results
    in a new pytree with the same structure. Non-tensor leaves are copied.
    """
    # Stopping condition
    if isinstance(tree, (Number, th.Tensor)):
        return func(cast(LeafInput, tree))

    # Recursive case
    if isinstance(tree, dict):
        return {k: pytree_map(func, cast(LeafInput, v)) for k, v in tree.items()}

    if isinstance(tree, list):
        return [pytree_map(func, cast(LeafInput, v)) for v in tree]

    if isinstance(tree, tuple):
        return tuple(pytree_map(func, cast(LeafInput, v)) for v in tree)

    if strict:
        raise TypeError(
            f"Found leaf '{tree}' of unsupported type '{type(tree).__name__}',"
            f" use `strict=False` to ignore"
        )
    else:
        return tree
