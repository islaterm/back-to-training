import torch
from torch import Tensor


# region : Activation functions
def sig(t: Tensor) -> Tensor:
    return torch.reciprocal(1 + torch.exp(-1 * t))


def tanh(t: Tensor) -> Tensor:
    exp_t = torch.exp(t)
    exp_neg_t = torch.exp(-1 * t)
    return (exp_t - exp_neg_t) * torch.reciprocal(exp_t + exp_neg_t)


def relu(t: Tensor) -> Tensor:
    """ Rectifier activation function.
        The relu function of a tensor T is the element-wise max between 0 and the appropriate 
        element of T.
    """
    tensor = t if torch.is_tensor(t) else torch.tensor(t)
    return torch.max(tensor, torch.zeros_like(tensor))


def swish(t: Tensor, beta: float) -> Tensor:
    """ Swish activation function proposed by Ramachandran et al. on their paper "Searching for
        Activation Functions" (arXiv:1710.05941v2).
        The Swish function of a tensor T is defined as: T * sigmoid(beta * T).
    """
    tensor = t if torch.is_tensor(t) else torch.tensor(t)
    beta_tensor = torch.full_like(tensor, beta)
    return tensor * sig(beta_tensor * tensor)


def celu(t: Tensor, alpha: float) -> Tensor:
    """ Continuously Differentiable Exponential Linear Units function as proposed by Barron on his
        paper "Continuously Differentiable Exponential Linear Units" (arXiv:1704.07483).
        The CELU function of a tensor T is:
            - T[i] when T[i] >= 0
            - alpha * (exp(T[i] / alpha) - 1)
        for each element i of the tensor T.
    """
    tensor = t if torch.is_tensor(t) else torch.tensor(t)
    zero_tensor = torch.zeros_like(tensor)
    alpha_tensor = torch.full_like(tensor, alpha)
    return torch.max(zero_tensor, tensor) + torch.min(zero_tensor, alpha_tensor * (
            torch.exp(tensor / alpha_tensor) - torch.full_like(tensor, 1)))


def softmax(t: Tensor, dim: int, stable=True) -> Tensor:
    """ Softmax function.
        The function stabilizes the values of a sequence of real values by generating a new
        sequence: S = exp(X) / sum(exp(*X)).
    """
    if stable:
        t -= t.max(dim=dim, keepdim=True)[0]
    exp = torch.exp(t)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


# endregion
# region : Derivatives
def d_dx_sigmoid(x: Tensor) -> Tensor:
    """Derivative of the sigmoid function."""
    return torch.ones_like(x) - sig(x)


def d_dx_tanh(x: Tensor) -> Tensor:
    """d/dx tanh(x)"""
    return torch.pow(torch.square(torch.cosh(x)), -1)


def d_dx_relu(x: Tensor) -> Tensor:
    """d/dx ReLU(x)"""
    return torch.where(torch.tensor(x >= 0), torch.ones_like(x), torch.zeros_like(x))


def d_dx_swish(x: Tensor, beta: float) -> Tensor:
    """d/dx swish(x, beta)"""
    return sig(torch.mul(beta, x))


def d_db_swish(x: Tensor, beta: float) -> Tensor:
    """d/d(beta) swish(x, beta)"""
    sig_bx = sig(torch.mul(beta, x))
    return torch.square(x) * sig_bx * (torch.ones_like(sig_bx) - sig_bx)


def d_dx_celu(x: Tensor, alpha: float) -> Tensor:
    """d/dx CELU(x, alpha)"""
    return torch.where(torch.tensor(x >= 0), torch.ones_like(x), torch.exp(torch.div(x, alpha)))


def d_da_celu(x: Tensor, alpha: float) -> Tensor:
    """d/d(alpha) CELU(x, alpha)"""
    return torch.where(torch.tensor(x >= 0), torch.zeros_like(x), torch.mul(-1, torch.div(
        torch.mul(x, torch.exp(torch.div(x, alpha))), alpha)) + torch.exp(
        torch.div(x, alpha)) - torch.ones_like(x))

# endregion
