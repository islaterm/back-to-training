from timeit import default_timer as timer
from typing import Callable, Dict, List, Union

import torch
from torch import Tensor
from torch.nn import Module

import activation_functions as ac_fn
from autocorrect import corrector, token

ActivationFunction = Union[Callable[[torch.Tensor, float], Tensor], Callable[[Tensor], Tensor]]


# noinspection PyPep8Naming
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
             l_a_params]) if l_a_params is not None else [[]] * len(l_a)
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
                torch.mm(out, weights) + biases, *params)
            self.__computed_layers.append(out)
        return ac_fn.softmax(torch.mm(out, self.__weights[-1]) + self.__biases[-1], dim=1)

    def backward(self, x: Tensor, y: Tensor, y_pred: Tensor) -> None:
        """
        Updates the network's parameters using the backpropagation algorithm.

        Args:
            x:
                the network's input.
            y:
                the expected output.
            y_pred:
                the actual output.
        """
        out: Tensor = self.__weights[-1]
        hidden_layers: List[Tensor] = self.__weights[1:-2]
        features: Tensor = self.__weights[0]

        batch_size = x.size()[0]
        # dL/du^{L + 1}
        dL_duLm1 = (y_pred - y) / batch_size
        # dL/dU
        out.grad = torch.t(
            self.__activation_functions[-1](self.__computed_layers[-1])) @ dL_duLm1
        # dL/dc
        self.__biases[1].grad = torch.sum(dL_duLm1, 0)
        # dL/dh_L
        dL_dhL = dL_duLm1 @ torch.t(out)

        derivatives: Dict[ActivationFunction, ActivationFunction]
        derivatives = { ac_fn.relu: ac_fn.d_dx_relu,
                        ac_fn.celu: ac_fn.d_dx_celu,
                        ac_fn.swish: ac_fn.d_dx_swish,
                        ac_fn.tanh: ac_fn.d_dx_tanh,
                        ac_fn.sig: ac_fn.d_dx_sigmoid }

        layer_idx = self.__size - 1
        while layer_idx >= 0:
            derivative = derivatives[self.__activation_functions[layer_idx + 1]]
            params = self.__function_parameters[layer_idx + 1]
            # dL/du^(k)
            dL_du_k = dL_dhL * derivative(self.__computed_layers[layer_idx + 1], *params)
            # dL/dW^(k) = dL/du^(k) * h^(k-1)
            hidden_layers[layer_idx].grad = torch.t(
                self.__activation_functions[layer_idx](self.__computed_layers[layer_idx])) @ dL_du_k
            # dL/db^(k) = dL/du^(k)
            # Note that we move the index in 2 because 0 is the first and 1 is the last
            self.__biases[layer_idx + 2].grad = torch.sum(dL_du_k, 0)
            # dL/dh^(k - 1) = dL/du^(k) * W^(k)
            dL_dhL = dL_du_k @ torch.t(hidden_layers[layer_idx])
            layer_idx -= 1

        # dL/du^(0) = dL/dh^(0) * d/dx(f(u^(0)))
        derivative = derivatives[self.__activation_functions[0]]
        params = self.__function_parameters[0]
        dL_du_0 = dL_dhL * derivative(self._cache[0], *params)
        # dL/dW^(0) = dL/du^(0) * F
        features.grad = torch.t(x) @ dL_du_0  # el anterior es h[-1] pero ese es el x
        # dL/db^(k) = dL/du^(k)
        self.__biases[0].grad = torch.sum(dL_du_0, 0)

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
        self.__biases = torch.nn.ParameterList(
            [torch.nn.Parameter(b) for b in biases + [classes]])

    @property
    def weights(self):
        return self.__weights.parameters()

    @property
    def in_size(self):
        return self.__weights[0].shape[0]
    # endregion


if __name__ == '__main__':
    # Tests del API del curso
    # (estos Tests NO sustituyen al anterior en la verificación de los gradientes)
    for test in ['mnist-model']:
        # Obtenemos los parámetos de la red desde el API
        F, l_h, l_a, C, Ws, U, bs, c, X, y = corrector.get_test_data(homework=2, question="3a",
                                                                     test=test, token=token)
        l_a = [f for s in l_a for f in [ac_fn.sig, ac_fn.tanh, ac_fn.relu, ac_fn.celu] if
               f.__name__ == s]

        # Inicializamos modelo con parámetros del API
        your_model = FFNN(F=F, l_h=l_h, l_a=l_a, C=C)
        your_model.load_weights([torch.Tensor(l) for l in Ws], torch.Tensor(U),
                                [torch.Tensor(l) for l in bs], torch.Tensor(c))

        # Obtenemos el índice del parámetro Ws[1] en la lista de parámetros de tu modelo
        idx = next(i for i, p in enumerate(your_model.parameters()) if
                   p.size() == torch.Tensor(Ws[1]).size() and torch.all(torch.Tensor(Ws[1]) == p))

        # Ejecutemos el forward de para input del API
        y_pred = your_model(torch.Tensor(X))

        # Ejecutemos el backward de tu modelo para ver como se comporta
        s = timer()
        your_model.backward(torch.Tensor(X), torch.Tensor(y), y_pred)
        t = timer() - s

        # Veamos todo fue OK :)
        # Si el Test te falla algunas veces por [time], puedes hacer time=0 ;-)
        corrector.sumbit(homework=2, question="3a", test=test, token=token,
                         answer=list(your_model.parameters())[idx].grad.mean(), time=t)
