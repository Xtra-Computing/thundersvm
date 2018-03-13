Introduction
======
In this page, we present Support Vector Machines (SVMs) and the Sequential Minimumal Optimazation (SMO) solver for training SVMs.

## Support Vector Machines (SVMs)

SVMs have been used in various applications including spam filtering, document classification, network attack detection. SVMs have good generalization property via maximizing margin of the seperating hyperplane. We discuss SVMs for binary classification here, because other SVM training problems such as SVM regression and ``$ \nu $``-SVMs can be converted into binary SVM training problems. The figure below shows an example of binary SVMs which have an optimal hyperplane seperating the training data set denoted by circles and squares. For non-linearly seperable data, SVMs use the kernel functions to map the data to a higher dimentional data space and also allows errors by introducing slack variables (see the *where* under the optimization problem below).
<center>![optimal hyperplan](https://docs.opencv.org/2.4/_images/optimal-hyperplane.png)</center>

In the following, we describe the formal defination of SVMs. Specifically, a training instance ``$ x_i $`` is attached with an integer ``$ y_i \in \{+1, -1\} $`` as its label. A positive (negative) instance is a training instance with the label of +1 (-1). Given a set ``$ \mathcal{X} $`` of n training instances, the goal of training SVMs is to find a hyperplane that separates the positive and the negative instances in ``$ \mathcal{X} $`` with the maximum margin and meanwhile, with the minimum misclassification error on the training instances. The training is equivalent to solving the following optimization problem:
```math
& \underset{\boldsymbol{w}, \boldsymbol{\xi}, b}{\text{argmin}}
& \frac{1}{2}{||\boldsymbol{w}||^2} + C\sum_{i=1}^{n}{\xi_i}\\
& \text{subject to}
&  y_i(\boldsymbol{w}\cdot \boldsymbol{x}_i - b) \geq 1 - \xi_i \\
& & \xi_i \geq 0, \ \forall i \in \{1,...,n\}
```
where ``$ \boldsymbol{w} $`` is the normal vector of the hyperplane, C is the penalty parameter, ``$ \boldsymbol{\xi} $`` is the slack variables to tolerant some training instances falling in the wrong side of the hyperplane, and b is the bias of the hyperplane.

To handle the non-linearly separable data, SVMs use a mapping function to map the training instances from the original data space to a higher dimensional data space where the data may become linearly separable. The optimization problem above can be rewritten to a dual form where the dot products of two mapped data can be replaced by a kernel function which avoids explicitly defining the mapping functions as only dot products are involved. The optimization problem in the dual form is shown as follows.
```math
& \underset{\boldsymbol{\alpha}}{\text{max}}
& & F(\boldsymbol{\alpha})=\sum_{i=1}^{n}{\alpha_i}-\frac{1}{2}{\boldsymbol{\alpha^T} \boldsymbol{Q} \boldsymbol{\alpha}}\\
& \text{subject to}
& &  0 \leq \alpha_i \leq C, \forall i \in \{1,...,n\}\\
& & & \sum_{i=1}^{n}{y_i\alpha_i} = 0
```
where ``$ F(\boldsymbol{\alpha}) $`` is the objective function; ``$ \boldsymbol{\alpha} \in \mathbb{R}^n $`` is a weight vector, where ``$ \alpha_i $`` denotes the _weight_ of ``$ \boldsymbol{x}_i $``; C is the penalty parameter; ``$ \boldsymbol{Q} $`` is a positive semi-definite matrix, where ``$ \boldsymbol{Q} = [Q_{ij}] $``, ``$ Q_{ij} = y_i y_j K(\boldsymbol{x}_i, \boldsymbol{x}_j) $`` and ``$ K(\boldsymbol{x}_i, \boldsymbol{x}_j) $`` is a kernel value computed from a kernel function (e.g., Gaussian kernel, ``$ K(\boldsymbol{x}_i, \boldsymbol{x}_j) = exp\{-\gamma||\boldsymbol{x}_i-\boldsymbol{x}_j||^2\} $``). All the kernel values together form an ``$ n \times n $`` kernel matrix.

## Sequential Minimal Optimization (SMO)

The goal of the training is to find a weight vector ``$ \boldsymbol{\alpha} $`` that maximizes the value of the objective function ``$ F(\boldsymbol{\alpha}) $``. Here, we describe a popular training algorithm, the Sequential Minimal Optimization (SMO) algorithm. It iteratively improves the weight vector until the optimal condition of the SVM is met. The optimal condition is reflected by an _optimality indicator vector_ ``$ \boldsymbol{f} = \langle f_1, f_2, ..., f_n \rangle $`` where ``$ f_i $`` is the optimality indicator for the i-th instance ``$ \boldsymbol{x}_i $`` and ``$ f_i $`` can be obtained using the following equation:``$ f_i = \sum_{j=1}^{n}{\alpha_j y_j K(\boldsymbol{x}_i, \boldsymbol{x}_j) - y_i} $``. In each iteration, the SMO algorithm has the following three steps:

**Step 1**: Search two extreme instances, denoted by ``$ \boldsymbol{x}_{u} $`` and ``$ \boldsymbol{x}_{l} $``, which have the maximum and minimum optimality indicators, respectively. It has been proven that the indexes of ``$ \boldsymbol{x}_{u} $`` and ``$ \boldsymbol{x}_{l} $``, denoted by u and l respectively, can be computed by the following equations.
```math
u = \text{argmin}_{i}\{f_i| \boldsymbol{x}_i \in \mathcal{X}_{upper}\}\\
l = \text{argmax}_{i}\{\frac{(f_{u} - f_i)^2}{\eta_i} | f_{u}<f_i, \boldsymbol{x}_i \in \mathcal{X}_{lower}\}
```
where
```math
\mathcal{X}_{upper} = \mathcal{X}_1 \cup \mathcal{X}_2 \cup \mathcal{X}_3,\\
\mathcal{X}_{lower} = \mathcal{X}_1 \cup \mathcal{X}_4 \cup \mathcal{X}_5;\\
\text{and}\\
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

### Prediction
After the training, the trained SVM is used to predict the label of unseen instances. The label of an instance ``$ \boldsymbol{x}_j $``, denoted by ``$ y_j $``, is predicted by the following formula:
```math
y_j = \text{sgn} (\sum_{i=1}^{n}y_i\alpha_iK(\boldsymbol{x}_i, \boldsymbol{x}_j) + b)
```
where b is the bias of the hyperplane of the trained SVM. The training instances with their weights greater than zero are called _support vectors_, which are used to predict the labels of unseen instances.

## Other SVM Training Problems
The other SVM training problems implemented in ThunderSVM can be modeled as binary SVM training problems. More specifically,

* SVM regression: 
The SVM regression training problem can be modeled as a binary SVM training problem, where each instance in the data set is duplicated, such that each instance is associated with two new training instances: one with label of +1 and the other with label of -1.
* Multi-class SVM classification: 
Multi-class SVM classsification problem can be decomposed into a few binary SVM training problems via pair-wise coupling (also known as one-vs-one decomposition).
* Probabilistic SVMs: 
Training probabilistic SVMs can be modeled as training binary SVMs and then use the decision values of the binary SVMs to fit a sigmoid function in order to obtain probabilities.
* ``$ \nu $``-SVMs: 
Training ``$ \nu $``-SVMs is also very similar to training binary SVMs. The key difference is that instead of using two training instances to the currently trained model, ``$ \nu $``-SVMs use four training instances. Training ``$ \nu $``-SVMs for regression (``$ \nu $``-SVR) is similar to training SVMs for regression.