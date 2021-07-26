**2021-7-20**

# CS-231n L-4,5

Morning: KNN assignment finished (without cross-validation)

## Intro to Neural Network

$$
s=f(x;W)=Wx\\
L_i=\sum_{j\not=y_i}\max(0,s_j-s_{y_i}+1)\\
L=1/N\sum_{i=1}^NL_i+\sum_{k}W_k^2
$$

- First how to define a classifier using a function $f$ which parametrized by weights $W$
- SVM loss function
- total loss = data loss + regularization (how simple our model is)

Once we construct our computation graph, we can apply the technique called backpropagation which is going to recursively using chain rule to compute the gradient with respect to every variable

.**Convnet/ Alexnet/Turing neural machine**

#### Backpropagation

- Compute the 'local' gradients
- Use the Chain Rule
- Receiving the value from upstream and transferring backwards the value after multiplying

![image-20210720141958620](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210720141958620.png)

*Within the computation graph, we can define the computational nodes at any granularity that we want to. eg.sigmoid gate*

$$\sigma(x)=\dfrac{1}{1+e^{-x}}$$

$$\dfrac{d\sigma}{dx}=\sigma(1-\sigma)$$

#### Patterns in backward flow

- **add** gate: gradient distributor (all the same)
- **max** gate: gradient router
- **mul** gate: gradient switcher (scale it with the value of the other branch)

##### A vectorized example

$f(x,W)=||w\cdot x||^2=\sum_{i=1}^{n}(W\cdot x)^2_i$

Note: always check the gradient with respect to a variable should have the same shape as the variable.

Kronecker Delta --- Indicator function

### Modularized implementation: forward/backward API

Caffe is a deep learning framework

How to understand the forward pass and backward pass?

### Summary

- neural nets will be very large: impractical to write down gradient formula by hand for all parameters
- **backpropagation**= recursive application of the chain rule along a computational graph to compute the gradients of all inputs/parameters/intermediates
- maintain a graph structure where the nodes implement the forward() and backward() API
- **forward**: compute result of an operation and save any intermediates needed for gradient computation in memory
- **backward**: apply the chain rule to compute the gradient of the loss function with respect to the inputs

### Neural Networks

Linear score function -> 2-layer neural network

$f=Wx\to f=W_2\max(0,W_1x) \to f=W_3\max(0,W_2\max(0,W_1x))$

That is where deep neural networks coming from

![image-20210720160822212](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210720160822212.png)

- We arrange neurons into fully-connected layers
- The abstraction of a layer has the nice property that it allows us to use efficient vectorized code

## Convolutional neural networks

A bit of history

- **Topographical mapping in the cortex:** nearby cells in cortex represent nearby regions in the visual field
- Neurocognitron
- Modern incarnation of CNN

**ConvNets** are everywhere!!!

- classification
- detection
- segmentation
- image retrieval
- video & pose & face recognition/ street signs /galaxies/diagnosis of medical images
- image captioning

#### Convolution Layer

- Convolve the filter with the image i.e. "slide over the image spatially, computing dot products"
- filters always extend the full depth of the input volume
- we are convolving with the flipped version of the filter
- this is related to convolution of two signals (elementwise multiplication and sum of a filter and the signal/ image)

Dimension of input --- N                        **Output size:** $(N-F)/stride+1$

Filter size --- F

In practice, we zero pad the borders (increase the input dimension by 2)

- Q1: Input volume is 32\*32\*3 10 5*5 filters with stride 1, pad 2, what is the output size?
- Output volume size: (32+4-5)/1+1=32 spatially, so 32\*32\*10
- Q2: Number of parameters in this layer?
- each filter has $5*5*3+1=76$ params (1 is for the bias term) $76*10=760$

**Number of the filters will be width of the output**

![image-20210720174438361](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210720174438361.png)

#### Pooling layer

- makes the representations smaller and more manageable
- operates over each activation map independently
- downsampling (spatially)
- MAX pooling is commonly used

#### Fully connected layer (FC layer)

- contains neurons that connect to the entire input volume, as in ordinary Neural Networks

![image-20210720182122649](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210720182122649.png)

