

 **2021-7-26**

Task 1: understand and consume how Adam works?

### Adam optimization algorithm

$$
v_{dw}=0,S_{dw}=0. v_{db}=0, S_{db}=0
$$

On iteration t:

​     Compute dW, db using current mini-batch

$v_{dw}=\beta_1v_{dw}+(1-\beta_1)dW,v_{db}=\beta_1v_{db}+(1-\beta_1)db\\S_{dw}=\beta_2v_{dw}+(1-\beta_2)dW^2, S_{db}=\beta_2v_{db}+(1-\beta_2)db^2$​
$$
V_{dw}^{corrected}=v_{dw}/(1-\beta^t_1), V_{db}^{corrected}=v_{db}/(1-\beta^t_1)\\
S_{dw}^{corrected}=S_{dw}/(1-\beta^t_2), S_{db}^{corrected}=S_{db}/(1-\beta^t_2)
$$
Finally, we have
$$
w=w-\alpha\dfrac{v_{dw}^{corrected}}{\sqrt{s_{dw}^{corrected}+\epsilon}}\\
b=b-\alpha\dfrac{v_{db}^{corrected}}{\sqrt{s_{db}^{corrected}+\epsilon}}
$$

#### Hyperparameters choice:

- $\alpha$: needs to be tuned
- $\beta_1$: 0.9
- $\beta_2$: 0.999
- $\epsilon$: 10-e8
- Adam: Adaptive moment estimation

# Continue: Training the Neural Network II

### Second-Order Optimization

(1). Use gradient and **Hessian** to form **quadratic** approximation

(2). Step to the **minima** of the approximation

$J(\theta)\approx J(\theta_0)+(\theta-\theta_0)^T\nabla_{\theta}J(\theta_0)+1/2(\theta-\theta_0)^TH(\theta_0)(\theta-\theta_0)$

We extrapolate from 1d to multi-variate function

$x_{critical}=x_0-\dfrac{f'(x_0)}{f''(x_0)}$

**To**

$\theta^*=\theta_0-H^{-1}\nabla J(\theta_0)$

**Q: What is nice about this update?**

Ans: there is no need for learning rate cuz once we derive the second-order approximation, the critical point is easy to obtain.

