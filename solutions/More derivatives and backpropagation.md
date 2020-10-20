## Last layer derivative

Let $S(T)$ be the softmax function of a tensor $T$.
The derivative of $S$ is given by:
$$
  \begin{aligned}
    \frac{\partial}{\partial T_j} S_i(T_j)
      &= \frac{\partial}{\partial T_j} \frac{e^{T_i}}{\sum_k e^{T_k}} \\
      &= \begin{cases}
        \frac{e^{T_i}\sum_k e^{T_k} - e^{T_j} e^{T_i}}{(\sum e^{t_k})^2} &\text{ if } i = j  \\
        \frac{0 - e^{T_j} e^{T_i}}{(\sum e^{t_k})^2} &\text{ otherwise}
      \end{cases} \\
      &= \begin{cases}
        \frac{e^{T_i}\left(\sum_k e^{T_k} - e^{T_j}\right)}{(\sum e^{t_k})^2} &\text{ if } i = j  \\
        \frac{e^{T_j}}{\sum e^{t_k}}\frac{e^{T_i}}{\sum e^{t_k}} &\text{ otherwise}
      \end{cases} \\
      &= \begin{cases}
        S_i(T_j)(1 - S_j(T_j)) &\text{ if } i = j  \\
        -S_i(T_j) \cdot S_j(T_j) &\text{ otherwise}
      \end{cases}
  \end{aligned}
$$

Now we can compute the derivative of the last layer as:
$$
  \begin{aligned}
    \frac{\partial \mathcal{L}}{\partial u^{(L + 1)}}(y, \hat{y})
      &= - \frac{1}{N} \frac{\partial}{\partial u^{(L + 1)}} \sum_i y_i \log \hat{y}_i  \\
      &= - \frac{1}{N} \sum_i y_i  \frac{\partial}{\partial u^{(L + 1)}} \log \hat{y}_i \\
      &= - \frac{1}{N} \sum_i y_i  \frac{\partial}{\partial \hat{y}_i} \log \hat{y}_i
        \frac{\partial}{\partial u^{(L + 1)}} S_i\left(u^{(L + 1)}\right) \\
      &= - \frac{1}{N} \sum_i y_i  \frac{1}{\hat{y}_i}
        \frac{\partial}{\partial u^{(L + 1)}} S_i\left(u^{(L + 1)}\right) \\
      &= - \frac{1}{N} \left(
          y(1 - \hat{y}) - \sum_i y_i \frac{1}{\hat{y}_i} (-\hat{y}_i \cdot \hat{y})
        \right) \\
      &= - \frac{1}{N} \left(y + y \cdot \hat{y} + \sum_i y_i \hat{y}\right) \\
      &= \frac{1}{N} \left(\hat {y} \left(y + \sum_i y_i\right) - y \right) \\
      &= \frac{1}{N} (\hat{y} - y)
  \end{aligned}
$$

## Last layer derivative (cont.)

Considering that $u^{(L + 1)} = h^{(L)} U + c$ and $\hat{y} = S(u^{(L + 1)})$.
We can compute the following derivatives:

$$
  \begin{aligned}
    \frac{\partial \mathcal{L}}{\partial U}
      &= \frac{\partial \mathcal{L}}{\partial u^{(L + 1)}}
        \cdot \frac{\partial u^{(L + 1)}}{\partial U} \\
      &= \frac{\partial \mathcal{L}}{\partial u^{(L + 1)}} 
        \cdot \frac{\partial}{\partial U} (h^{(L)} U + c) \\
      &= \frac{\partial \mathcal{L}}{\partial u^{(L + 1)}} h^{(L)}
  \end{aligned}
$$
