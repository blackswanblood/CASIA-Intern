**2021-7-22**

# cs231n - Training the neural networks II

Machine learning def

- *A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured in by P, improves with experience E*

#### Recap Weight Initialization

- Too small
  - Activations go to zero, gradients also zero. No learning
- Too big
  - Activations saturate, gradients will also turn to 0. No learning
- Just right
  - Nice distribution of activations at all layers. Learning proceeds nicely!

#### Batch Normalization

In order to accelerate Deep Network Training by Reducing Internal Covariate Shift

Coarse range to start with for all the hyperparameters

### Overview

- Fancier optimization
- Regularization
- Transfer Learning

### Optimization

Problems with SGD

1. What if loss changes quickly in one direction and slowly in another?
2. What does gradient descent do?
3. Behavior at local minima and saddle points (Saddle points much more common than local minima)

Ans: Very slow progress along shallow dimension, jitter along steep direction

#### SGD + Momentum

$$
v_{t+1}=\rho v_t+\nabla f(x_t)   \\
$$

$$
x_{t+1}=x_t-\alpha v_{t+1}
$$

```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x += learning_rate * vx
```

- Build up "velocity" as a running mean of gradients
- Rho gives "friction"; typically rho = 0.9

#### Nesterov Momentum

$v_{t+1}=\rho v_t-\alpha\nabla f(x_t+\rho v_t)$

$x_{t+1}=x_t+v_{t+1}$

#### AdaGrad

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared += dx*dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
# Added element-wise scaling of the gradient based on the historical sum of squares in each dimension
```

#### RMSProp

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared = decay_rate * grad_squared + (1- decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)

```

### Adam (FULL)

```python
first_moment = 0
second_moment = 0
while True:
    dx = compute_gradient(x)
    first_moment = beta1*first_moment + (1-beta1) * dx
    second_moment = beta2 * second_moment + (1- beta2)* dx * dx
    first_unbias = first_moment / (1-beta1**t)
    second_unbias = second_moment / (1-beta2**t)  # Bias term
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)

```

Another optimization method is called learning rate decay (with SGD + Momentum)!
