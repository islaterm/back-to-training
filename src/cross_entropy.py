import torch
from torch import Tensor


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


def cross_entropy_loss_derivative(t: Tensor, t_pred: Tensor) -> Tensor:
    """ Calculates the derivative of the CELoss function."""
    return 1 / t.size()[0] * (t_pred - t)


if __name__ == '__main__':
    B, C = 5, 10
    y = torch.ones(B, C)
    y_pred = torch.ones(B, C)
    dimL = 40
    hL = torch.ones(B, dimL)
    U = torch.ones(dimL, C)
    c = torch.ones(C)
    uLm1 = hL @ U + c

    dL_duLm1 = cross_entropy_loss_derivative(y, y_pred)

    assert dL_duLm1.size() == uLm1.size()
