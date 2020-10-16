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
