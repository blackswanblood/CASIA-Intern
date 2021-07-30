**2021-7-27**

Morning: Get ready for the code in assignment 1

​              ： Familiarize myself with Deep learning framework

# Continue: CS-231n L8

- Can use an **optimizer** to compute gradients and update weights
- Remember to execute the output of the optimizer

The process of receiving an input to produce some kind of output to make some kind of prediction is known as Feed Forward.

In the **feed-forward** neural network (aka Multilayer perceptron), there are not any feedback loops or connections in the network. It is a directed acyclic Graph which means that there are no feedback connections or loops in the network. It has an input layer, an output layer, and a hidden layer. In general, there can be multiple hidden layers.

### Keras: High-level wrapper

- Keras is a layer on top of TensorFlow
- Supports Theano backend

![image-20210727100841349](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210727100841349.png)

### PyTorch: three Levels of abstraction (Equivalency in tf)

- **Tensor**: Imperative ndarray, but runs on GPU (Numpy array)
- **Variable:** Node in a computational graph; stores data and gradient (Tensor, variable, placeholder)
- **Module:** A neural network layer; may store state or learnable weights (tf.layers, or TFSlim, TFLearn or Sonnet) 

#### DataLoaders

A **DataLoader** wraps a Dataset and provides minibatching, shuffling, multithreading, for you

When you need to load custom data, just write your own Dataset class

**Tensorboard vs Visdom**

### Static vs Dynamic Graphs

- **TensorFlow:** Build graph once, then run many times (static)
- **pyTorch:** Each forward pass defines a new graph (**Dynamic**)

With static graphs, framework can **optimize** the graph for you before it runs!

### Static vs Dynamic Serialization

**Static**

- Once graph is built, can **serialize** it and run it without the code that built the graph!

**Dynamic**

- Graph building and execution are intertwined, so always need to keep code around

- TensorFlow Fold make dynamic graphs easier in tensorFlow through **Dynamic Batching**

### Dynamic Graph Application

- Recurrent networks
- recursive networks
- Modular Networks (neuralmodule network)

### Caffe

- Core written in c++
- Has python and MATLAB bindings
- Good for training or finetuning feedforward classification models
- Often no need to write code!
- Not used as much in research anymore, still popular for deploying models

### Summary

**TensorFlow** is a safe bet for most projects. Not perfect but has huge community, wide usage. Maybe pair with high-level wrapper (Sonnet)

**pytorch** is best for research. However still new, there can be rough patches.

Use **TensorFlow** for one graph over many machines

**Caffe, Caffe2 TF** for production deployment

**Tf or Caffe2** for mobile

### PCA [classic algorithm that retains the most info of dataset when dimension deduction]



- - 
