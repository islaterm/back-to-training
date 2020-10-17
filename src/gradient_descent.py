from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


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


class RandomDataset(Dataset):
    """A dataset initialized with random values."""
    __input_features: Tensor
    __expected_classification: Tensor

    def __init__(self, n_examples: int, n_features: int, n_classes: int):
        """
        Initializes a dataset with random values.

        Args:
            n_examples:
                the number of examples in the dataset.
            n_features:
                the number of features per example.
            n_classes:
                the number of classes on the target function.
        """
        self.__input_features = torch.randn(n_examples, n_features)
        one_hot_classes = torch.eye(n_classes)
        self.__expected_classification = \
            one_hot_classes[torch.randint(n_classes, (1, n_examples)), :].view(-1, n_classes)

    def __len__(self) -> int:
        """Returns the size of the dataset."""
        return self.__input_features.size()[0]

    def __getitem__(self, i) -> Tuple[Tensor, Tensor]:
        """Returns an entry on the database."""
        return self.__input_features, self.__expected_classification


if __name__ == '__main__':
    N, F, C = 100, 300, 10
    your_dataset = RandomDataset(N, F, C)

    print("Correct Test!" if len(your_dataset) == N else "Failed Test [len]")
    print("Correct Test!" if type(your_dataset[N // 2]) == tuple and len(
        your_dataset[N // 3]) == 2 else "Failed Test [getitem]")
