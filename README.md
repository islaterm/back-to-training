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

#### (b) Entropía Cruzada

Comenzaremos haciendo una función para computar la pérdida de nuestra red.
Recuerda que para dos distribuciones de probabilidad discreta $p(x)$ y $q(x)$ la entropía cruzada
(cross entropy) entre $p$ y $q$ está dada por
$$
  \it{CE}(p, q) = \sum_{x} p(x) \log\left(\frac{1}{q(x)}\right) = -\sum_{x} p(x) \log q(x)
$$
donde $x$ varía sobre todos los posibles valores para los cuales la distribución está definida.

En esta parte debes programar la función `CELoss` que recibe tensores $\mathbf{Q}_{ij}$ y
$\mathbf{P}_{ij}$ (de las mismas dimensiones) y calcula el promedio de las entropías cruzadas de
las distribuciones $p_i$ y $q_i$ de la siguiente forma
$$
  \mathit{CELoss}(\mathbf{Q}, \mathbf{P}) = \frac{1}{N} \sum_{i}\mathit{CE}(p_{i}, q_{i})
$$
donde $p_i(x) = \mathbf{P}_{ix}$, $q_i(x) = \mathbf{Q}_{ix}$ y $N$ es el tamaño de la primera
dimension de los tensores (dimension `0`).
Nota que el resultado es un escalar.
Nota también el orden de $\mathbf{Q}$ y $\mathbf{P}$ en $\mathit{CELoss}(\mathbf{Q}, \mathbf{P})$.
Esto no es un error, sino es la forma standard de usar la entropía cruzada como una función de
error, en donde el primer argumento ($\mathbf{Q}$) es la aproximación (distribución de probabilidad
erronea) y el segundo argumento ($\mathbf{P}$) es el valor al que nos queremos acercar
(distribución de probabilidad real, o más percisamente en nuestro caso, distribución de
probabilidad empírica).

En nuestra implementación debemos evitar cualquier ocurrencia de `NaN` debido a valores en nuestras
distribuciones de probabilidad excesivamente pequeños al calcular `torch.log`.
Estos valores deberían devolver números negativos demasiado pequeños para procesar y dan como
resultado `NaN`.
El valor épsilon limitará el valor mínimo del valor original cuando `estable=True`.

### Parte 2: Más derivadas y back propagation

En esta parte comenzaremos a usar el algoritmo de back propagation para poder actualizar los
parámetros de nuestra red neuronal (la que empezaste a construir en la Tarea 1).
Nuestra red está dada por las ecuaciones
$$
  \begin{aligned}
    h^{(\ell)}  & = f^{(\ell)}(h^{(\ell - 1)} W^{(\ell)} + b^{(\ell)}) \\
    \hat{y}     & = \mathit{softmax}(h^{(L)}U + c).
  \end{aligned}
$$

Recuerda que en estas ecuaciones consideramos que el $h^{(0)}$ es el tensor de input, digamos $x$,
y típicamente llamamos a $\hat{y}$ como $\hat{y} = \mathit{forward}(x)$.

Para optimizar los parámetros de nuestra red usaremos la función de pérdida/error de entropía 
cruzada (ver la parte anterior).
Dado un conjunto (mini batch) de ejemplos $\{(x_1, y_1), \ldots, (x_B, y_B)\}$, llamemos $x$ al
tensor que contiene a todos los $x_i$'s *apilados* en su dimensión $0$.
Nota que $x$ tendrá una dimensión más que los $x_i$'s.
Similarmente llamemos $y$ al tensor que contiene a todos los $y_i$'s. La función de pérdida de la
red se puede entonces escribir como
$$
  \mathcal{L} = \mathit{CELoss}(\hat{y}, {y})
$$
donde $\hat{y} = \mathit{forward}(x)$ y $\mathit{CELoss}(\hat{y},{y})$ es la función de entropía
cruzada aplicada a $\hat{y}$ e $y$.
En esta parte computaremos las derivadas parciales
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

### Parte 4: Descenso de gradiente y entrenamiento

En esta parte programaras el algoritmo de descenso de gradiente más común y entrenarás finalmente tu
red para que encuentre parámetros que le permitan clasificar datos aleatorios (mas abajo podrás
hacerlo opcionalmente también para MNIST).

#### (a) Descenso de gradiente (estocástico)

Construye una clase `SGD` que implemente el algoritmo de descenso de gradiente. El inicializador de
la clase debe recibir al menos dos argumentos: un "iterable" de parámetros a los cuales aplicarles
el descenso de gradiente, y un valor real `lr` correspondiente a la taza de aprendizaje para el
descenso de gradiente.
El único método que debes implementar es el método `step` que debe actualizar todos los parámetros.
En este caso asumiremos que a cada parámetro ya se le han computado los gradientes (todos
almacenados en el atributo `.grad` de cada parámetro).
El uso de esta clase debiera ser como  sigue:

```python
# datos = iterador sobre pares de tensores x, y
# red = objeto FFNN previamente inicializado

optimizador = SGD(red.parameters(), 0.001)
for x,y in datos:
  y_pred = red.forward(x)
  l = CELoss(y_pred, y)
  red.backward(x, y, y_pred)
  optimizador.step()
```

#### (b) Datos para carga

En esta parte crearás un conjunto de datos de prueba aleatorios para probar con tu red.
La idea de partir con datos al azar es para que te puedas concentrar en encontrar posibles bugs en
tu implementación antes de probar tu red con cosas más complicadas.

Para esta parte debes crear una clase `RandomDataset` como subclase de `Dataset` (que se encuentra
en el módulo `torch.utils.data`).
Tu clase debe recibir en su inicializador la cantidad de ejemplos a crear, la cantidad de
características de cada ejemplo, y la cantidad de clases en la función objetivo.
Debes definir la función `__len__` que retorna el largo del dataset y la función `__getitem__` que
permite acceder a un item específico de los datos.
Cada elemento entregado por `__getitem__` debe ser un par $(x, y)$ con un único ejemplo, donde $x$
es un tensor que representa a los datos de entrada (características) e $y$ representa al valor
esperado de la clasificación para esa entrada.

Lo positivo de definir un conjunto de datos como `Dataset` es que luego puedes usar un `DataLoader`
para iterar por paquetes sobre el dataset y entregarlos a una red (tal como lo hiciste en la Tarea 1
para MNIST).
El siguiente trozo de código de ejemplo muestra cómo debieras usar tu clase en conjunto con un
`DataLoader`.

```python
dataset = RandomDataset(1000, 200, 10)
data = DataLoader(dataset, batch_size=4)
for x,y in data:
  # x,y son paquetes de 4 ejemplos del dataset.
```
