from timeit import default_timer as timer
from typing import Callable, Union

import torch
from torch import Tensor, nn

from autocorrect import corrector, token

ActivationFunction = Union[Callable[[torch.Tensor, float], Tensor], Callable[[Tensor], Tensor]]


# Acá solo deberías programar la función backward.
# El resto del código viene de la Tarea 1 (a menos que hayas programado
# la parte opcional en cuyo caso también deberías cambiar el inicializador).
# Puedes incluir todo el código de la Tarea 1 que quieras.

# Funciones Tarea 1
def sig(T):
    return torch.reciprocal(1 + torch.exp(-1 * T))


def tanh(T):
    E = torch.exp(T)
    e = torch.exp(-1 * T)
    return (E - e) * torch.reciprocal(E + e)


def relu(T):
    return torch.clamp(T, min=0.0)


def swish(T, B=1):
    return T * sig(B * T)


def celu(T, A=1):
    # max(0,x)+min(0,α∗(exp(x/α)−1))
    e = A * torch.expm1(T / A)
    M = torch.clamp(T, min=0.0)
    m = torch.clamp(e, max=0.0)
    return m + M


def softmax(T, dim, estable=True):
    if estable:
        T -= torch.max(T, dim, True)[0]
    Ei = torch.exp(T)
    Si = torch.sum(Ei, dim, True)
    return Ei / Si


# Funciones nuevas

def derivadaFn(fn, u):
    alfa = 1
    beta = 1
    derivada = 0
    if fn == celu:
        ones = torch.ones_like(u)
        derivada = torch.where(u >= 0, ones, torch.exp(u / alfa))

    elif fn == relu:
        ones = torch.ones_like(u)
        zeros = torch.zeros_like(u)
        derivada = torch.where(u >= 0, ones, zeros)

    elif fn == swish:
        derivada = sig(beta * u) * (beta * u * (1 - sig(beta * u)) + 1)

    elif fn == sig:
        derivada = sig(u) * (1 - sig(u))

    elif fn == tanh:
        derivada = 1 - (tanh(u) * tanh(u))

    return derivada


# CLASE FFNN

class FFNN(torch.nn.Module):
    # código desde la Tarea 1
    def __init__(self, F, l_h, l_a, C):
        super(FFNN, self).__init__()
        self.__in = nn.Parameter(torch.randn(F, l_h[0]))
        self.__hiddens = nn.ParameterList(
            [nn.Parameter(torch.randn(l_h[i], l_h[i + 1])) for i in range(len(l_h) - 1)])
        self.__out = nn.Parameter(torch.randn(l_h[-1], C))
        self.__bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(i.size()[1])) for i in self.parameters()])
        self.__fn = l_a

    # código desde la Tarea 1, editado
    def forward(self, x):

        # h = self.__fn[0](x.mm(self.__in)+self.__bias[0])

        u = x.mm(self.__in) + self.__bias[0]
        self._cache = [u]
        h = self.__fn[0](u)

        # for i in range(len(self.__hiddens)):
        #  h = self.__fn[i+1](h.mm(self.__hiddens[i])+self.__bias[i+2])
        # y = softmax(h.mm(self.__out)+self.__bias[1],1)
        # return y

        for i in range(len(self.__hiddens)):
            u = h.mm(self.__hiddens[i]) + self.__bias[i + 2]
            self._cache.append(u)
            h = self.__fn[i + 1](u)

        y = softmax(h.mm(self.__out) + self.__bias[1], 1)
        return y

    def getin(self):
        return self.__in

    def gethiddens(self):
        return self.__hiddens

    def getout(self):
        return self.__out

    def getbias(self):
        return self.__bias

    def getfn(self):
        return self.__fn

    def setin(self, ins):
        self.__in.data = ins.data

    def sethiddens(self, hiddens):
        for i in range(len(hiddens)):
            self.__hiddens[i].data = hiddens[i].data

    def setout(self, out):
        self.__out.data = out.data

    def setbias(self, bias):
        for i in range(len(bias)):
            self.__bias[i].data = bias[i].data

    def setfn(self, fn):
        self.__fn = fn

    # .data = los datos sean igual a los datos

    # nuevo código Tarea 2
    def backward(self, x, y, y_pred):
        # computar acá todos los gradientes

        # tamaño del batch
        b = x.size()[0]

        # el final se hace aparte, partimos de atras para adelante

        i = len(self.__hiddens) - 1  # el -1 va pq los indices parten en 0

        # dL_du_{L+1} = uwu
        dL_duLm1 = (1 / b) * (y_pred - y)  # El +1 es pq es el ultimo

        # dL_dU = dL_du_{L+1} y hL
        self.__out.grad = self.__fn[-1](
            self._cache[-1]).t() @ dL_duLm1  # lo de la izquierda es el h[i]
        # (la ultima funcion aplicada al ultimo u)
        # dL_dc = dL_du_{L+1}
        self.__bias[1].grad = torch.sum(dL_duLm1, 0)  # El último bias se guarda en la posición 1

        # dL_dhL = dL_du_{L+1} * U
        dL_dhL = dL_duLm1 @ self.__out.t()

        while (i >= 0):  # se recorre al revez porque es backward

            # dL_duk = dL_dhk * derivada(fn(uk))
            dL_duk = dL_dhL * derivadaFn(self.__fn[i + 1], self._cache[i + 1])

            # dL_dWk = dL_duk * h_{k-1}
            self.__hiddens[i].grad = self.__fn[i](self._cache[i]).t() @ dL_duk  # El anterior es i

            # dL_dbk = dL_duk
            self.__bias[i + 2].grad = torch.sum(dL_duk,
                                                0)  # El +2 va porque el 0 es el primero y el 1
            # es el ultimo, los demás van en los hiddens
            # dL_dh_{k-1} = dL_duk * Wk
            dL_dhL = dL_duk @ self.__hiddens[i].t()  # idem, es el anterior

            i -= 1

        # Ahora la primera capa!
        # pq ibamos al verres

        # dL_du0 = dL_dh0 * derivada(fn(u0))
        dL_duk = dL_dhL * derivadaFn(self.__fn[0], self._cache[0])

        # dL_dW0 = dL_du0 * h_{-1?}
        self.__in.grad = x.t() @ dL_duk  # el anterior es h[-1] pero ese es el x

        # dL_dbk = dL_duk
        self.__bias[0].grad = torch.sum(dL_duk, 0)

        # En teoria ahi asignamos todos los gradientes creo... es todo lo que hay que hacer aca

    def num_parameters(self):
        total = 0
        for p in self.parameters():
            total += p.numel()
        return total


if __name__ == '__main__':
    # Tests del API del curso
    # (estos Tests NO sustituyen al anterior en la verificación de los gradientes)
    for test in ['mnist-model']:
        # Obtenemos los parámetos de la red desde el API
        F, l_h, l_a, C, Ws, U, bs, c, X, y = corrector.get_test_data(homework=2, question="3a",
                                                                     test=test, token=token)
        l_a = [f for s in l_a for f in [sig, tanh, relu, celu] if f.__name__ == s]

        # Inicializamos modelo con parámetros del API
        your_model = FFNN(F=F, l_h=l_h, l_a=l_a, C=C)
        your_model.setin(torch.tensor(Ws[0]))
        your_model.sethiddens([torch.tensor(w) for w in Ws[1:-2]])
        your_model.setout(torch.tensor(U))
        your_model.setbias([torch.tensor(b) for b in bs + [c]])

        # Obtenemos el índice del parámetro Ws[1] en la lista de parámetros de tu modelo
        # idx = next(i for i, p in enumerate(your_model.parameters()) if
        #            p.size() == torch.Tensor(Ws[1]).size() and torch.all(torch.Tensor(Ws[1]) == p))

        # Ejecutemos el forward de para input del API
        y_pred = your_model(torch.Tensor(X))

        # Ejecutemos el backward de tu modelo para ver como se comporta
        s = timer()
        your_model.backward(torch.Tensor(X), torch.Tensor(y), y_pred)
        t = timer() - s

        # Veamos todo fue OK :)
        # Si el Test te falla algunas veces por [time], puedes hacer time=0 ;-)
        corrector.sumbit(homework=2, question="3a", test=test, token=token,
                         answer=your_model.gethiddens()[0].grad.mean(), time=t)
