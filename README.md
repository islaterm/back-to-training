# Back to training

![http://creativecommons.org/licenses/by/4.0/](https://i.creativecommons.org/l/by/4.0/88x31.png)

This work is licensed under a
[Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)

## Instructions as-is

En esta tarea programarás backpropagation para el caso específico de las redes que construiste en la
[Tarea 1](https://colab.research.google.com/drive/1aeuSRjj_kQ_uFEBSJ9bRuyr4G4MY4FAi) (si tuviste
problemas resolviendo la Tarea 1 puedes usar
[esta solución tipo](https://colab.research.google.com/drive/1whxzPx0jBRu2v1GD-s_VhhYS-w3Tlu9E) que
preparó el cuerpo docente).
Además comenzarás a entrenar la red con descenso de gradiente, que también tendrás que programar.

Te recomendamos que repases la materia de las clases de backpropagation. En particular para resolver
esta tarea te puedes apoyar en el siguiente material:

* [Clase 04-2020: Descenso de Gradiente para encontrar los parámetros de una red.](https://www.youtube.com/watch?v=G4dnRSSC6Kw)
* [Clase 05-2020: Introducción a Backpropagation](https://www.youtube.com/watch?v=1EUAoM1EhM0)
* [Clase 06-2020: Continuación Backpropagation](https://www.youtube.com/watch?v=Gp2rY7LvTyQ)

Algunos videos de versiones pasadas del curso que te pueden servir también de apoyo son los
siguientes (los actualizaremos con la versión 2020 cuando los tengamos disponibles):

* [Entropia Cruzada, funcion de pérdida y tensores (2018)](https://www.youtube.com/watch?v=lnYAVf1UkU8)
* [Derivando tensores y backpropagation a mano (2018)](https://www.youtube.com/watch?v=atQHDde309k)

IMPORTANTE: A menos que se exprese lo contrario, sólo podrás utilizar las clases y funciones en el
módulo [`torch`](https://pytorch.org/docs/stable/torch.html).

(por Jorge Pérez, [jorgeperezrojas](https://github.com/jorgeperezrojas),
[@perez](https://twitter.com/perez))

### Parte 1: Preliminares: funciones de activación y función de error

#### (a) Derivando las funciones de activación

En esta parte debes calcular a mano las derivadas de las funciones `relu`, `swish` y `celu` que usamos en la [Tarea 1](https://colab.research.google.com/drive/1aeuSRjj_kQ_uFEBSJ9bRuyr4G4MY4FAi). Recuerda que `swish` y `celu` tienen ambas parámetros adicionales así que debes calcular las derivadas (parciales) con respecto a ellos también. Intenta expresar las derivadas en términos de aplicaciones de la misma función (o sub expresiones de esta). Por ejemplo, si derivas la función $\text{sigmoid}(x)$ (hazlo! es un buen ejercicio) encontrarás que su derivada se puede expresar como:

$$
  \frac{\partial\ \text{sigmoid}(x)}{\partial x} = \text{sigmoid}(x)\big(1 - \text{sigmoid}(x)\big)
$$

Usa la
[Hoja de respuesta](https://colab.research.google.com/drive/1a44G8JIfuaAXmare28dCDT1gvUV1CuDP) para incluir tus expresiones.

### Parte 2: Más derivadas y back propagation

En esta parte comenzaremos a usar el algoritmo de back propagation para poder actualizar los parámetros de nuestra red neuronal (la que empezaste a construir en la Tarea 1). Nuestra red está dada por las ecuaciones
$$
\begin{aligned}
  h^{(\ell)}  & = f^{(\ell)}(h^{(\ell - 1)} W^{(\ell)} + b^{(\ell)}) \\
  \hat{y}     & = \mathit{softmax}(h^{(L)}U + c).
\end{aligned}
$$

Recuerda que en estas ecuaciones consideramos que el $h^{(0)}$ es el tensor de input, digamos $x$, y típicamente llamamos a $\hat{y}$ como $\hat{y}=\mathit{forward}(x)$.

Para optimizar los parámetros de nuestra red usaremos la función de pérdida/error de entropía cruzada (ver la parte anterior). Dado un conjunto (mini batch) de ejemplos $\{(x_1,y_1),\ldots,(x_B,y_B)\}$, llamemos $x$ al tensor que contiene a todos los $x_i$'s *apilados* en su dimensión $0$. Nota que $x$ tendrá una dimensión más que los $x_i$'s. Similarmente llamemos $y$ al tensor que contiene a todos los $y_i$'s. La función de pérdida de la red se puede entonces escribir como

$$
\mathcal{L} = \mathit{CELoss}(\hat{y}, {y})
$$
donde $\hat{y}=\mathit{forward}(x)$ y $\mathit{CELoss}(\hat{y},{y})$ es la función de entropía cruzada aplicada a $\hat{y}$ e $y$. En esta parte computaremos las derivadas parciales

$$
\frac{\partial \mathcal{L}}{\partial \theta}
$$
para cada parámetro $\theta$ de nuestra red.

#### (a) Derivando la última capa

Recuerda que $\hat y = \mathit{softmax}(h^{(L)}U + c)$.
Nuestro objetivo en esta parte es calcular la derivada de $\mathcal{L}$ con respecto a $U$,
$h^{(L)}$ y $c$.
Para esto llamemos primero

$$
u^{(L+1)} = h^{(L)}U + c.
$$

Nota que con esto, nuestra predicción es simplemente $\hat{y} = \mathit{softmax}(u^{(L + 1)})$.
Calcula la derivada (el *gradiente*) de $\mathcal{L}$ respecto a $u^{(L + 1)}$, y escribe un trozo
de código usando las funcionalidades de `torch` que calcule el valor y lo almacene en una variable
`dL_duLm1`, suponiendo que cuentas con los tensores `y` e `y_pred` (que representa a $\hat{y}$).
