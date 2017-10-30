Introduction
======
## Support Vector Machines

## Sequential Minimal Optimization
A training instance ``$ x_i $`` is attached with an integer ``$ y_i \in \{+1, -1\} $`` as its label. A positive (negative) instance is a training instance with the label of +1 (-1). Given a set ``$ \mathcal{X} $`` of n training instances, the goal of training SVMs is to find a hyperplane that separates the positive and the negative instances in ``$ \mathcal{X} $`` with the maximum margin and meanwhile, with the minimum misclassification error on the training instances. The training is equivalent to solving the following optimization problem:
```math
& \underset{\boldsymbol{w}, \boldsymbol{\xi}, b}{\text{argmin}}
& \frac{1}{2}{||\boldsymbol{w}||^2} + C\sum_{i=1}^{n}{\xi_i}\\
& \text{subject to}
&  y_i(\boldsymbol{w}\cdot \boldsymbol{x}_i - b) \geq 1 - \xi_i \\
& & \xi_i \geq 0, \ \forall i \in \{1,...,n\}
```
where ``$ \boldsymbol{w} $`` is the normal vector of the hyperplane, C is the penalty parameter, ``$ \boldsymbol{\xi} $`` is the slack variables to tolerant some training instances falling in the wrong side of the hyperplane, and b is the bias of the hyperplane.

To handle the non-linearly separable data, SVMs use a mapping function to map the training instances from the original data space to a higher dimensional data space where the data may become linearly separable. The optimization problem above can be rewritten to a dual form where mapping functions can be replaced by kernel functions which make the mapping easier. The optimization problem in the dual form is shown as follows.
```math
& \underset{\boldsymbol{\alpha}}{\text{max}}
& & F(\boldsymbol{\alpha})=\sum_{i=1}^{n}{\alpha_i}-\frac{1}{2}{\boldsymbol{\alpha^T} \boldsymbol{Q} \boldsymbol{\alpha}}\\
& \text{subject to}
& &  0 \leq \alpha_i \leq C, \forall i \in \{1,...,n\}\\
& & & \sum_{i=1}^{n}{y_i\alpha_i} = 0
```
where ``$ F(\boldsymbol{\alpha}) $`` is the objective function; ``$ \boldsymbol{\alpha} \in \mathbb{R}^n $`` is a weight vector, where ``$ \alpha_i $`` denotes the _weight_ of ``$ \boldsymbol{x}_i $``; C is the penalty parameter; ``$ \boldsymbol{Q} $`` is a positive semi-definite matrix, where ``$ \boldsymbol{Q} = [Q_{ij}] $``, ``$ Q_{ij} = y_i y_j K(\boldsymbol{x}_i, \boldsymbol{x}_j) $`` and ``$ K(\boldsymbol{x}_i, \boldsymbol{x}_j) $`` is a kernel value computed from a kernel function (e.g., Gaussian kernel, ``$ K(\boldsymbol{x}_i, \boldsymbol{x}_j) = exp\{-\gamma||\boldsymbol{x}_i-\boldsymbol{x}_j||^2\} $``). All the kernel values together form an ``$ n \times n $`` kernel matrix.

The goal of the training translates to finding a weight vector ``$ \boldsymbol{\alpha} $`` that maximizes the value of the objective function ``$ F(\boldsymbol{\alpha}) $``. Here, we describe a popular training algorithm, the Sequential Minimal Optimization (SMO) algorithm. It iteratively improves the weight vector until the optimal condition of the SVM is met. The optimal condition is reflected by an _optimality indicator vector_ ``$ \boldsymbol{f} = \langle f_1, f_2, ..., f_n \rangle $`` where ``$ f_i $`` is the optimality indicator for the i-th instance ``$ \boldsymbol{x}_i $`` and ``$ f_i $`` can be obtained using the following equation:``$ f_i = \sum_{j=1}^{n}{\alpha_j y_j K(\boldsymbol{x}_i, \boldsymbol{x}_j) - y_i} $``. In each iteration, the SMO algorithm has the following three steps:

**Step 1**: Search two extreme instances, denoted by ``$ \boldsymbol{x}_{u} $`` and ``$ \boldsymbol{x}_{l} $``, which have the maximum and minimum optimality indicators, respectively. It has been proven~\cite{keerthi2001improvements, fan2005working} that the indexes of ``$ \boldsymbol{x}_{u} $`` and ``$ \boldsymbol{x}_{l} $``, denoted by u and l respectively, can be computed by the following equations.
```math
u = \argminl_{i}\{f_i| \boldsymbol{x}_i \in \mathcal{X}_{upper}\}\\
l = \argmaxl_{i}\{\frac{(f_{u} - f_i)^2}{\eta_i} | f_{u}<f_i, \boldsymbol{x}_i \in \mathcal{X}_{lower}\}
```
where
```math
\mathcal{X}_{upper} = \mathcal{X}_1 \cup \mathcal{X}_2 \cup \mathcal{X}_3,\\
\mathcal{X}_{lower} = \mathcal{X}_1 \cup \mathcal{X}_4 \cup \mathcal{X}_5;\\
and\\
\mathcal{X}_{1} = \{\boldsymbol{x}_i| \boldsymbol{x}_i \in \mathcal{X}, 0 < \alpha_i < C\},\\
\mathcal{X}_{2} = \{\boldsymbol{x}_i| \boldsymbol{x}_i \in \mathcal{X}, y_i = +1, \alpha_i = 0\},\\
\mathcal{X}_{3} = \{\boldsymbol{x}_i| \boldsymbol{x}_i \in \mathcal{X}, y_i = -1, \alpha_i = C\},\\
\mathcal{X}_{4} = \{\boldsymbol{x}_i| \boldsymbol{x}_i \in \mathcal{X}, y_i = +1, \alpha_i = C\},\\
\mathcal{X}_{5} = \{\boldsymbol{x}_i| \boldsymbol{x}_i \in \mathcal{X}, y_i = -1, \alpha_i = 0\};
```
and ``$ \eta_i = K(\boldsymbol{x}_{u}, \boldsymbol{x}_{u}) + K(\boldsymbol{x}_{i}, \boldsymbol{x}_{i}) - 2K(\boldsymbol{x}_{u}, \boldsymbol{x}_{i}) $``; ``$ f_{u} $`` and ``$ f_{l} $`` denote the optimality indicators of ``$ \boldsymbol{x}_{u} $`` and ``$ \boldsymbol{x}_{l} $``, respectively.

**Step 2**: Improve the weights of ``$ \boldsymbol{x}_{u} $`` and ``$ \boldsymbol{x}_{l} $``, denoted by ``$ \alpha_{u} $`` and ``$ \alpha_{l} $``.
```math
\alpha_{l}' = \alpha_{l} + \frac{y_{l}(f_{u} - f_{l})}{\eta}\\
\alpha_{u}' = \alpha_{u} + y_{l} y_{u}(\alpha_{l} - \alpha_{l}')
```
where ``$ \eta = K(\boldsymbol{x}_{u}, \boldsymbol{x}_{u}) + K(\boldsymbol{x}_{l}, \boldsymbol{x}_{l}) - 2K(\boldsymbol{x}_{u}, \boldsymbol{x}_{l}) $``. To guarantee the update is valid, when ``$ \alpha_{u}' $`` or ``$ \alpha_{l}' $`` exceeds the domain of [0, C], ``$ \alpha_{u}' $`` and ``$ \alpha_{l}' $`` are adjusted into the domain.

**Step 3**: Update the optimality indicators of all instances. The optimality indicator ``$ f_i $`` of the instance ``$ \boldsymbol{x}_i $`` is updated to ``$ f'_i $`` using the following formula:
```math
f_i' = f_i + (\alpha_{u}' - \alpha_{u})y_{u} K(\boldsymbol{x}_{u}, \boldsymbol{x}_i)\\
   +\ (\alpha_{l}' - \alpha_{l}) y_{l} K(\boldsymbol{x}_{l}, \boldsymbol{x}_i)
```
SMO repeats the above steps until the optimal condition is met, i.e., ``$ f_{u} \ge f_{max} $``, where
```math
f_{max} = max\{f_i | \boldsymbol{x}_i \in \mathcal{X}_{lower}\}
```
After the optimal condition is met, we obtain the ``$ \boldsymbol{\alpha} $`` values which corresponding to the optimal hyperplane and the SVM with these ``$ \boldsymbol{\alpha} $`` values is considered trained.
