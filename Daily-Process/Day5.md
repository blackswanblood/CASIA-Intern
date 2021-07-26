**2021-7-23**

Morning: Game (Multi-agent game deep reinforcement learning) symposium

Realization of SVM, SGD



# Intro to RL - L1

### Course objective: Democratize RL

### What is reinforcement learning and why we care

![image-20210723175739621](C:\Users\fyx\AppData\Roaming\Typora\typora-user-images\image-20210723175739621.png)

> a computational approach to learning whereby an agent tries to maximize the total amount of <u>reward</u> it receives while interacting with a complex and uncertain environment.

### Supervised learning: (image classification)

- Annotated images, data follows iid distribution.
- Learners are told what the labels are.

### Reinforcement Learning (features)

- Data are not iid. Instead, a correlated time series data
- No instant feedback or label for correct action
- action (breakout): left right
- Trial-and-error exploration
- Delayed reward
- Time matters (sequential data)
- Agent's actions affect the subsequent data it receives (action affects the environment)

**Differences**:

- Sequential data as input (iid)
- The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them.
- Trial-and-error exploration (exploration and exploitation)
- There is no supervisor, only a reward signal, which is also delayed

#### Deep reinforcement learning

- analogy to traditional CV and deep CV
- end-to-end training

### Why RL works ?

- computation power boost: GPUs
- Acquire the high degree of proficiency in domains governed by simple, known rules
- End-to-end training, features and policy are jointly optimized toward the end goal