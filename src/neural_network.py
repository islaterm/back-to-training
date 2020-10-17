from typing import Callable, List, Union

import torch
from torch import Tensor
from torch.nn import Module

from activation_functions import celu, relu, softmax, swish

ActivationFunction = Union[Callable[[torch.Tensor, float], Tensor], Callable[[Tensor], Tensor]]


class FFNN(Module):
    """ Implementation of a Feed Forward Neural Network (FFNN)."""
    __computed_layers: [Tensor]

    def __init__(self, F: int, l_h: List[int], l_a: List[ActivationFunction], C: int,
                 l_a_params=None):
        """
        Creates a new feed forward neural network.

        Args:
            F:
                the number of neurons in the input layer.
            l_h:
                the list of sizes of the hidden layers.
            l_a:
                the list of the activation functions.
            C:
                the number of neurons on the output layer.
            l_a_params:
                the parameters for the activation functions
        """
        super(FFNN, self).__init__()

        sizes = [F] + l_h + [C]
        self.__weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(sizes[i], sizes[i + 1])) for i in
             range(len(sizes) - 1)])
        self.__biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(h)) for h in sizes[1:]])
        self.__activation_functions = l_a
        self.__function_parameters = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(p)) if p else None for p in
             l_a_params]) if l_a_params is not None else [None] * len(l_a)
        self.__computed_layers = []

    def forward(self, nn_input: Tensor) -> Tensor:
        """
        Performs a forward feeding to the network.

        Args:
            nn_input:
                the input of the network.
        Returns:
            the output of the feeding normalized with the softmax function.
        """
        out = nn_input
        self.__computed_layers = [nn_input]
        for weights, biases, activation, params in zip(self.__weights[:-1], self.__biases[:-1],
                                                       self.__activation_functions,
                                                       self.__function_parameters):
            out = activation(torch.mm(out, weights) + biases) if params is None else activation(
                torch.mm(out, weights) + biases, params.item())
            self.__computed_layers.append(out)
        return softmax(torch.mm(out, self.__weights[-1]) + self.__biases[-1], dim=1)

    def backward(self, x, y, y_pred):
        """"""
        it = self.__size - 1
        batch_size = x.size()[0]
        out_layer_loss = (y_pred - y) / batch_size
        out: Tensor = self.__weights[-1]
        out.grad = self.__activation_functions[-1]
        # TODO

    # region : Utility
    def summary(self) -> None:
        """
        Prints a summary of the weights of this network.
        """
        for name, p in self.named_parameters():
            print(f'{name}:\t{p.size()}')

    # endregion

    # region : Properties
    def load_weights(self, weights: Tensor, outputs: Tensor, biases: Tensor, classes: Tensor):
        self.__weights = torch.nn.ParameterList(
            [torch.nn.Parameter(W) for W in weights + [outputs]])
        self.__biases = torch.nn.ParameterList([torch.nn.Parameter(b) for b in biases + [classes]])

    @property
    def weights(self):
        return self.__weights.parameters()

    @property
    def in_size(self):
        return self.__weights[0].shape[0]
    # endregion


if __name__ == '__main__':
    model = FFNN(300, [50, 30, 25, 20], [relu, celu, swish, relu], 10, [None, .5, .5, None])
    print(model.summary())
    print(model.in_size)
    print(model(torch.rand(2, 300)))
