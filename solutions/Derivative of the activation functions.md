# Derivative of the activation functions

## *ReLU*

$$
  \frac{\partial}{\partial x}\mathit{ReLU}(x) =
    \frac{\partial}{\partial x}\max\{0, x\} =
    \begin{cases}
      1 & \text{if } x > 0  \\
      0 & \text{otherwise}
    \end{cases}
$$

## *Swish*

Considerando que
$$
  \frac{\partial}{\partial x}\sigma(x) = 1 - \sigma(x)
$$

Se tiene
$$
  \begin{aligned}
    \frac{\partial}{\partial x}\mathit{swish}(x, \beta)
      &= \frac{\partial}{\partial x}(x \cdot \sigma(\beta x))  \\
      &= \frac{\partial x}{\partial x}\sigma(\beta x)
        + x \frac{\partial}{\partial x}\sigma(\beta x)  \\
      &= \sigma(\beta x)
        + x \frac{\partial}{\partial x}\sigma(x)\frac{\partial}{\partial x} \beta x  \\
      &= \sigma(\beta x) + \beta x (1 - \sigma(x))  \\
  \end{aligned}
$$

$$
  \begin{aligned}
    \frac{\partial}{\partial \beta}\mathit{swish}(x, \beta)
      &= \frac{\partial}{\partial \beta}(x \cdot \sigma(\beta x)) \\
      &= \frac{\partial x}{\partial \beta} \sigma(\beta x)
        + x\frac{\partial}{\partial \beta}\sigma(\beta x)  \\
      &= \frac{\partial x}{\partial \beta} \sigma(\beta x)
        + x \frac{\partial}{\partial (\beta x)}\sigma(\beta x)
          \frac{\partial}{\partial \beta} \beta x \\
      &= x^2 \sigma(\beta x)(1 - \sigma(\beta x))
  \end{aligned}
$$

$$
\frac{\partial\ \text{celu}(x, \ldots)}{\partial x} = \ldots
$$

$$
\frac{\partial\ \text{celu}(x, \ldots)}{\partial \ldots} = \ldots
$$
