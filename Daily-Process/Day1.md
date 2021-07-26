**2021-7-19**

# CS-231n



### L1- Intro to convolutional neural networks for visual recognition



- First phd thesis about CV
  - the visual world is simplified into simple geometric shapes
  - to recognize and reconstruct these shapes (blocks)
- Generalized cylinder and pictorial structure (simple shape objects)
- lines and edges (object recognition)



#### Image segmentation

#### Face detection 

#### SIFT

- feature based object recognition
- some features remain diagnostic and invariant to changes (viewpoint, occlusion)
- match the features to similar object
- holistic scene/ Spatial Pyramid Matching

*Potential problems for CV*

- Overfit the model during training process
- high complexity of the model tends to have a high dimension of input (loads of parameters to fit)

> convolutional neural network is the winner of ImageNet Challenge and we should take a deep dive into the realm of deep learning (CNN/Convnets)

- Object detection
- Action classification
- Image captioning



**Key innovations that renovate the classic algorithms**

- Computation /Moore's Law (GPUs - parallelizable)
- Data (large labelled data is needed)

*The quest for visual intelligence goes far beyond, understand the image in a rich and deep way*

- semantic segmentation
- perceptual grouping
- activity recognition



### L2 - Image Classification pipeline

description: A core task in computer vision

Problem: semantic gap between the image and the pixel values (RGBH), camera position, illumination, deform, occlusion, Clutter - robustness, intra class variation

```python
def classify_image(image):
    # !!! This is an image classifier
    return class_label
```

There is no obvious way to hard-code the algorithm for recognizing a cat, or other classes

**Data-driven approach**

1. Collect a dataset of images and labels
2. Use ML to train a classifier
3. Evaluate the classifier on new images

```python
def train(images, labels):
    return model
#ML
#KNN: memorize all data and labels
def predict(model, test_images):
    return test_labels
# Model to predict labels
# predict the label of the most similar training image
```

**Distance Metric to compare images**

L1 distance/Manhattan: $d_1(I_1,I_2)=\sum_{p}|I^{p}_1-I^p_2|$

*Ideal situation:*

- we want classifiers that are fast at prediction; slow for training is ok

L2 distance/ Euclidean: $d_2(I_1,I_2)=\sqrt{\sum_p(I^{p}_1-I^p_2)^2}$

k in knn is called **hyperparameters:** choices about the algorithm that we set rather than learn; this is very problem-dependent and we must try them all out and see what works best!

#### Setting hyperparameters

- split data into train, validation, and test; choose hyperparameters on val and evaluate on testing set
- **Cross-validation:** Split data into folds, try each fold as validation and average the results (this is useful for small datasets, but not used too frequently in deep learning)

### Summary of KNN:

- In image classification we start with a training set of images and labels, and must predict labels on the test set. 
- The KNN classifier predicts labels based on nearest training examples, 
- distance metric and K are hyperparameters
- Choose hyperparameters using the **validation set**; only run on the test set once at the very end!

#### Why knn is never used on images?

- inverse time length expectation for training and testing phase
- curse of dimensionality, exponential growth (not enough images to densely cover the space// In the high dimensional space)

### Linear regression

> A neural network can be viewed as a series of logistic regression classifiers stacked on top of each other.
>
> Deep neural networks are like legos, linear classifiers can be viewed as elementary building blocks

##### Parametric Approach

The first component of this approach is to define the score function that maps the pixel values of an image to confidence scores for each class. We will develop the approach with a concrete example. As before, let’s assume a training dataset of images $x_i∈R^Dx_i∈R^D$, each associated with a label yiyi. Here i=1…Ni=1…N and yi∈1…Kyi∈1…K. That is, we have **N** examples (each with a dimensionality **D**) and **K** distinct categories. For example, in CIFAR-10 we have a training set of **N** = 50,000 images, each with **D** = 32 x 32 x 3 = 3072 pixels, and **K** = 10, since there are 10 distinct classes (dog, cat, car, etc). We will now define the score function $f:R^D↦R^Kf:R^D↦R^K$ that maps the raw image pixels to class scores.

**Linear classifier.** In this module we will start out with arguably the simplest possible function, a linear mapping: (Learn a single template for each category)

$f(x_i,W,b)=Wx_i+bf(x_i,W,b)=Wx_i+b$

In the above equation, we are assuming that the image xixi has all of its pixels flattened out to a single column vector of shape [D x 1]. The matrix **W** (of size [K x D]), and the vector **b** (of size [K x 1]) are the **parameters** of the function. In CIFAR-10, xixi contains all pixels in the i-th image flattened into a single [3072 x 1] column, **W** is [10 x 3072] and **b** is [10 x 1], so 3072 numbers come into the function (the raw pixel values) and 10 numbers come out (the class scores). The parameters in **W** are often called the **weights**, and **b** is called the **bias vector** because it influences the output scores, but without interacting with the actual data xixi. However, you will often hear people use the terms *weights* and *parameters* interchangeably.

There are a few things to note:

- First, note that the single matrix multiplication WxiWxi is effectively evaluating 10 separate classifiers in parallel (one for each class), where each classifier is a row of **W**.
- Notice also that we think of the input data (xi,yi)(xi,yi) as given and fixed, but we have control over the setting of the parameters **W,b**. Our goal will be to set these in such way that the computed scores match the ground truth labels across the whole training set. We will go into much more detail about how this is done, but intuitively we wish that the correct class has a score that is higher than the scores of incorrect classes.
- An advantage of this approach is that the training data is used to learn the parameters **W,b**, but once the learning is complete we can discard the entire training set and only keep the learned parameters. That is because a new test image can be simply forwarded through the function and classified based on the computed scores.
- Lastly, note that classifying the test image involves a single matrix multiplication and addition, which is significantly faster than comparing a test image to all training images.

> Foreshadowing: Convolutional Neural Networks will map image pixels to scores exactly as shown above, but the mapping ( f ) will be more complex and will contain more parameters.



### L3 - Loss functions and Optimization

**Goal:**

- Define a loss function that quantifies our unhappiness with the scores across the training data.
- Come up with a way of efficiently finding the parameters that minimize the loss function.

A **Loss function** tells how good our current classifier is. Given a dataset of examples $\{(x_i,y_i)\}^N_{i=1}$ where $x_i$ is image and $y_i$ is label
$$
L=\dfrac{1}{N}\sum_iL_i(f(x_i,W),y_i)
$$

#### **Multi-class SVM loss**/Hinge loss

$$
L_i=\sum_{j\not=y_i}\max(0,s_j-s_{y_i}+1)
$$

where we use the shorthand for the scores vector $s=f(x_i,W)$

- Q3: At initialization W is small so all s is approximately 0, what is the loss?
- The number of all the classes minus 1, since we loop over all the incorrect classes and we have the margin 1 to count
- Q4: What if the sum was over all classes (including j = y_i)
- Loss will increase by 1 and all the sum of loss will be lower-bounded by 1
- Q5: What if we used mean instead of sum?
- IT is a rescaling strategy, it does not affect the score
- Q6: Suppose that we found a W st L=0. Is this W unique?
- NO! 2W will also incur L=0.

![image-20210719155436647](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210719155436647.png)

**Data loss:** Model predictions should match training data

**Regularization:** Model should be "simple", so it works on test data
$$
R(W)=\sum_k\sum_lW^2_{k,l}\ ->L2\\
R(W)=\sum_k\sum_l|W_{k,l}|\ ->L1\\
R(W)=\sum_k\sum_l\beta W^2_{k,l}+|W_{k,l}|\ ->\text{Elastic net}\\
$$

- Max norm regularization
- Dropout
- Batch normalization, stochastic depth

#### **Softmax Classifier** (Multinomial Logistic Regression)/cross-entropy loss

Scores = unnormalized log probabilities of the classes
$$
P(Y=k|X=x_i)=\dfrac{e^sk}{\sum_je^{s_j}}
$$
where $s=f(x_i;W)$

Want to maximize the log likelihood, or to minimize the loss function of the correct class:
$$
L_i=-\log P(Y=y_i|X=x_i)
$$
![image-20210719165814881](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210719165814881.png)

### Optimization

- Random search NO!
- stochastic gradient descent

- In summary:
  - Numerical gradient: approximate, slow, easy to write
  - Analytic gradient: exact, fast, error-prone
  - IN practice, always use analytic gradient, but check implementation with numerical gradient. This is called a gradient check.

```python
# GD
while True:
     weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += - step_size*weights_grad
    
# Minibatch
while True:
    data_batch = sample_training_data(data,256)
    weights_grad = evluate_gradient(loss_fn, data_batch,weights)
    weight += - step_size * weights_grad
```



# DL-C1

- History of Deep learning (hype of neural networks; ups and downs)

