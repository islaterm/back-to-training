import torch


class SGD:
    """Implementation of a Stochastic Gradient Descent algorithm."""
    __parameters: torch.nn.ParameterList
    __learning_rate: float

    def __init__(self, parameters: torch.nn.ParameterList, lr: float):
        """
        Initializes a new SGD object.

        Args:
            parameters:
                the parameters of the neural network to be optimized
            lr:
                the learning rate of the algorithm
        """
        self.__parameters = parameters
        self.__learning_rate = lr

    def step(self) -> None:
        """Updates the parameters according to their respective gradients."""
        param: torch.nn.Parameter
        for param in self.__parameters:
            param.data -= self.__learning_rate * param.grad
