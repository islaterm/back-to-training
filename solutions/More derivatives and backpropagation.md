# Last layer derivative

$$
\begin{aligned}
  \frac{\partial \mathcal{L}}{\partial u^{(L + 1)}}(y, \hat{y})
    &= \frac{\partial}{\partial u^{(L + 1)}} \mathit{CELoss}(y, \hat{y})  \\
    &= \frac{\partial}{\partial u^{(L + 1)}} \frac{1}{N} \sum_i \mathit{CE}(\hat{y}_i, y_i) \\
    &= - \frac{\partial}{\partial u^{(L + 1)}} \frac{1}{N}\sum_i \hat{y}_i \log{y_i} \\
    &= - \frac{\partial}{\partial u^{(L + 1)}}
      \frac{1}{N}\sum_i \mathit{softmax}_i(u^{(L + 1)}) \log{y_i} \\
    &= - \frac{\partial}{\partial u^{(L + 1)}}
      \frac{1}{N}\sum_i \mathit{softmax}_i(u^{(L + 1)}) \log{y_i} \\
    %&= \frac{1}{|y|}(\hat{y} - y)
\end{aligned}
$$
