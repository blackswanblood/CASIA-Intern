**2021-7-21**

Morning: derive the gradient in back propagation algorithm in the form of matrix multiplication.

Todo: finish the cross-validation part of assignment 1

# cs-231n L6

## Training Neural Networks I

The whole process of finding the lowest value of loss function is now called

### Mini-batch SGD 

loop:

1. **Sample** a batch of data
2. **Forward** prop it through the graph/network, get loss
3. **Backprop** to calculate the gradients
4. **Update** the parameters using the gradient

### Overview

- **One time setup**

  - activation functions, preprocessing, weight initialization, regularization, gradient checking

- **Training dynamics**

  - babysitting the learning process, parameter updates, hyperparameter optimization

- **Evaluation**

  - model ensembles

  

#### Activation function - sigmoid

$\sigma(x)=1/(1+e^{-x})$

- squashes numbers to range [0,1]
- Historically popular since they have nice interpretation as a saturating "firing rate"  of a neuron
- 3 problems
  - Saturated neurons "kill" the gradients
  - Sigmoid outputs are not zero-centered (Insufficient gradient update)
  - exp() is a bit compute expensive

![image-20210721073011016](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210721073011016.png)

$\tanh(x)$

- Squashes numbers to range [-1,1]
- zero centered
- still kills gradients when saturated

$\max(0,x)$

- Rectified Linear Unit
- Does not saturate +
- Computationally efficient
- Converges faster tan sigmoid/tanh in practice
- More biologically plausible than sigmoid
- NOT zero-centered output
- Negative path has a saturation
- People like to initialize ReLU neurons with slightly positive biases (0.01)

$f(x)=\max(0.01x,x)$

$f(x)=\max(\alpha x,x)$

- Leaky ReLU
  - Does not saturate
  - Computationally efficient
  - Converges faster than sigmoid/tanh
  - **will not die**
- Parametric ReLU

![image-20210721075120930](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210721075120930.png)

#### Maxout "Neuron"

$\max(w_1^Tx+b_1,w_2^Tx+b_2)$

- Does not have the basic form of dot product -> nonlinearity

- Generalizes ReLU and Leaky ReLU

- Linear Regime, no saturation, does not die

- (-) doubles the number of parameters/ neuron

  

**In practice:**

- Use ReLU. Be careful with your learning rates
- Try out tanh but don't expect much
- DONT use sigmoid

### Data preprocessing

- zero-centered the data
- normalized data (image NO)
- PCA (Principal Component Analysis) and whitening

In practice for Images: center only

- Subtract the mean image (AlexNet) (mean image = [32,32,3] array)
- Subtract per-channel mean (VGGNet) (mean along each channel = 3 numbers)

### Weight Initialization

- Q: what happens when W=0?
- All the neurons will do the same thing/ same operations
- First idea: **Small random numbers** (gaussian with 0 mean and 1e-2 standard deviation)

```python
w = 0.01* np.random.randn(D,H)
```

Works okay for small networks, but problems with deeper network

# Xavier Initialization

Last week, we discussed backpropagation and gradient descent for deep learning models. All deep learning optimization methods involve an initialization of the weight parameters.

Let’s explore the **[first visualization in this article](https://www.deeplearning.ai/ai-notes/initialization/index.html)** to gain some intuition on the effect of different initializations.

**What makes a good or bad initialization? How can different magnitudes of initializations lead to exploding and vanishing gradients?**

**If we initialize weights to all zeros or the same value, what problem arises?**

![img](https://cs230.stanford.edu/doks-theme/assets/images/section/4/viz.png)

The goal of Xavier Initialization is to initialize the weights such that the variance of the activations are the same across every layer. This constant variance helps prevent the gradient from exploding or vanishing.

To help derive our initialization values, we will make the following **simplifying assumptions**:

- Weights and inputs are centered at zero
- Weights and inputs are independent and identically distributed
- Biases are initialized as zeros
- We use the tanh() activation function, which is approximately linear with small inputs: Var(a[l])≈Var(z[l])Var(a[l])≈Var(z[l])

**Let’s derive Xavier Initialization now, step by step.**

Our full derivation gives us the following initialization rule, which we apply to all weights:

W[l]i,j=N(0,1n[l−1])Wi,j[l]=N(0,1n[l−1])

![img](https://cs230.stanford.edu/doks-theme/assets/images/section/5/proof.png)

Xavier initialization is designed to work well with tanh or sigmoid activation functions. For ReLU activations, look into He initialization, which follows a very similar derivation.

### Batch Normalization

*you want unit gaussian activations? Just make them so!*

-  Consider a batch of activations at some layer. To make each dimension unit gaussian, apply:

$$
\hat{x}^{(k)}=\dfrac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}
$$

BN --- Usually inserted after Fully Connected or Convolutional layers and before nonlinearity

And then allow the network to squash the range if it wants to:

$y^{(k)}=\gamma^{(k)}\hat{x}^{(k)}+\beta^{(k)}$

![image-20210721180427766](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210721180427766.png)

### Babysitting the learning process

Double check that the loss is reasonable!

- loss not going down: learning rate too low
- loss exploding: learning rate too high

### Hyperparameter optimization

Cross-validation strategy

coarse -> fine cross-validation in stages

- first, only a few epochs to get rough idea of what params work
- second, longer running time, finer search
- if the cost is ever > 3* original cost, break out early
- Random Search vs Grid Search
- network architecture
- learning rate, its decay schedule, update type
- regularization (L2/Dropout strength)



