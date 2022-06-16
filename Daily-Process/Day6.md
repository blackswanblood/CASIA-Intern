

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

**Q: why is this bad for deep learning?**

Ans: Hessian matrix has too many elements and inverting a matrix is an expensive operation.

### Quasi-Newton method

- **L-BFGS** (Limited memory BFGS)
  - Works very well in full batch, deterministic mode
  - Does not transfer well to mini-batch setting

### Beyond training error

- performance on unseen data
- **Model ensembles**
  - Train multiple independent models
  - At test time average their results

### Regularization to improve single-model proficiency

1. **Dropout** (for neural network)
   - In each forward pass, randomly set some neurons to zero probability (activation) of dropping is a hyperparameter; 0.5 is common
   - Forces the network to have a redundant representation
   - Prevents co-adaptation of features

2. Still Dropout, but at test time
   - Dropout makes our output random! $y=f_W(x,z)$
   - Want to "average out" the randomness at test-time
   - Local property: **At test time, multiply by dropout probability is enough**
3. **Data augmentation**
   - translation
   - rotation
   - stretching
   - shearing
   - lens distortions
4. **Color jitter**
   - Randomize the contrast and brightness
   - Complex version: Apply PCA to all [rgb] pixels in training set, Sample a "color offset" along principal component directions, then add offset to all pixels of a training image
5. DropConnect
6. Fractional Max Pooling
7. Batch Normalization
8. Stochastic Depth

#### Summary

- Training: add some kind of randomness $y=f_{W}(x,z)$
- Testing: average out randomness (or approximate)
- $y=f(x)=E_z[f(x,z)]=\int p(z)f(x,z)dz$

### Transfer Learning

*You need a lot of a data if you want to train/use CNNs* (Busted)

![image-20210726114824813](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210726114824813.png)

**Transfer learning with CNNs is pervasive** (norm)

1. Find a very large dataset that has similar data, train a big ConvNet there
2. Transfer learn to your dataset

**Deep learning frameworks**: Caffe, TensorFlow, PyTorch

## Summary:

- Optimization
  - Momentum, RMSprop, AdaGrad etc
- Regularization
  - Dropout
- Transfer learning

# Intro to RL II

### Agent and environment

- The agent learns to interact with the environment
- Agent imposes action upon the environment and the consequences are observation and reward

### Rewards

- A reward is a scalar feedback signal
- Indicate how well agent is doing at step t
- Reinforcement learning is based on the maximization of rewards

### Sequential Decision Making

- objective of the agent: select a series of actions to maximize total future rewards
- Actions may have long term consequences
- Reward may be delayed
- Trade-off between immediate reward and long-term reward
- The history is the sequence of observations,actions,rewards. $H_t=O_1,R_1,A_1,...,A_{t-1},O_t,R_t$
- State is the function used to determine what happens next $S_t=f(H_t)$​
- Environment state and agent state
- **Full observability:** agent directly observes the environment state, formally as Markov decision process (MDP) $O_t=S^e_t=S^a_t$
- **Partial observability:** agent indirectly observes the environment, formally as partially observable Markov decision process

### Major components of an RL agent

- Policy: agent's behavior function
- Value function: how good is each state or action
- Model: agent's state representation of the environment

#### Policy

- A policy is the agent's behavior model
- It is a map function from state/observation to action
- Stochastic policy: Probabilistic sample $\pi(a|s)=P[A_t=a|S_t=s]$
- Deterministic policy: $a^* =\arg\max_a\pi(a|s)$

#### Value function

- Value function: expected discounted sum of future rewards under a particular policy $\pi$
- Discount factor weights immediate vs future rewards
- Used to quantify goodness/badness of states and actions
- Q-function (could be used to select among actions)

#### Model

- A model predicts what the environment will do next
- predict the next state: $p = \mathbb{P}[S_{t+1}=s'|S_t=s,A_t=a]$
- predict the next reward $R=\mathbb{E}[R_{t+1}|S_t=s,A_t=a]$

![image-20210726143242976](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210726143242976.png)

### Types of RL Agents based on what the agent learns

- Value-based agent:
  - Explicit: Value function
  - Implicit: policy (can derive a policy from value function)
- Policy-based agent:
  - Explicit: policy
  - No value function
- Actor-Critic agent
  - Explicit: policy and value function



#### Whether there exists model

- Model-based
  - Explicit: model
  - May or may not have policy and/or value function
- Model-free
  - Explicit : value function and / or policy function
  - No model

[How to understand 'convolution' operation](https://www.zhihu.com/question/22298352)

# CS231n- L8 Deep learning frameworks

### Overview

- [CPU vs GPU](##CPU vs GPU)
- [Deep learning frameworks](##Deep learning frameworks)
  - Caffe/Caffe2
  - Theano/TensorFlow
  - Torch/PyTorch

## CPU vs GPU

![image-20210726171119448](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210726171119448.png)

- CPU cores can operate independent with each other and they are all fast and powerful
- GPU cores are slow, but they are great at parallelizable tasks

### Programming GPUs

- CUDA (NVIDIA only)
  - Write c-like code that runs directly on the GPU
  - Higher-level APIs: cuBLAS,cuFFT,cuDNN,etc
- OpenCL
  - Similar to CUDA, but runs on anything
  - slower
- Udacity: Intro to Parallel Programming cs344

### CPU/GPU Communication

If you aren't carful, training can bottleneck on reading data and transferring to GPU!

**Solution:**

- Read all data into RAM
- Use SSD instead of HDD
- Use multiple CPU threads to prefetch data

## Deep learning frameworks

1. easily build big computational graphs
2. easily compute gradients in computational graphs
3. run it all efficiently on GPU (wrap cuBLAS, cuDNN)

![image-20210726174647684](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210726174647684.png)

- First **define** computational graph
- Then **run** the graph many times
- can use an **Optimizer** to compute gradients and update weights 
