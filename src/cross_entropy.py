from timeit import default_timer as timer

import torch
from torch import Tensor

from autocorrect import corrector, token


# noinspection PyPep8Naming
def CELoss(Q: Tensor, P: Tensor, estable=True, epsilon=1e-8) -> float:
    """
    Cross entropy loss function.

    Args:
        Q:
            probability distribution tensor with the predicted values of the model.
        P:
            probability distribution tensor with the true (empirical) values of the model.
        estable:
            flag to stabilize the computation of small values.
        epsilon:
            limit of the minimum value.
            Used when estable=True.
    Returns:
        the average of the cross-entropy of the input tensors.
    """
    assert P.size() == Q.size()
    batch_size = P.size()[0]
    q = torch.clamp(Q, min=epsilon, max=1 - epsilon) if estable else Q
    return -1 / batch_size * torch.sum(P * torch.log(q))
