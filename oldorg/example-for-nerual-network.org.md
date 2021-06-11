# 神经网络

## 名称

一般约定，N-layers不包括输入层。因此，一个一层神经网络代表的一个没有隐藏层的神经
网络。在这种定义下，简单的逻辑回归或者支持向量机都是一个一层的神经网络。同时神经
网络也被称为人工神经网络(ANN, Artificial Neural
Networks)或者多层感应器(MLP, Multi-Layer Perceptrons).

## 组件

1.  节点，即神经元，位于层中
2.  权重，即链接线的粗细
3.  偏差，每一个节点都具有一个偏差
4.  激励函数，对层中每一个节点的输入进行转换，不同的层拥有不同的激励函数

## 大小

1.  如何计算神经网络中神经元的个数或者参数变量？神经元的个数有隐藏层所含的节点总
    数加上输出层节点个数，而参数变量为偏差+连接线。

2.  如何根据实际的问题决定层的个数？值得注意的是，随着神经网络的层数增加，网络可
    以完美地拟合出一条决策边界，但很容易导致过拟合问题。但过拟合问题可以使用其他
    方法防止。

3.  使用大网络带来的好处？小的网络很难使用梯度下降进行优化，梯度下降有个缺点就是
    可能收敛在局部极小值，而大的网络具有更多的极小值，而这些极小值可能比最小值具
    有更小的损失。

4.  小的网络输出的结果具有更大的波动，可能会收敛到一个很好的极小值，但也有可能会
    输出一个较差的结果。而在大的网络中，将会产生更多的结果，且最终的波动比小的网
    络更小。

    > n other words, all solutions are about equally as good, and rely
    > less on the luck of random initialization.

5.  因此，你不要因为过拟合问题而使用更小的网络，相反，如果条件允许，应该使用更大
    的网络以及使用正规惩罚等手段来控制过拟合问题。

## 层的类型

### 全链接层

故名思义，就是后一个层的每一个节点都与前一个层的所有节点相连。

### 输出层

输出层与其他的层不同，其没有激励函数，这是因为往往最后一层代表着类的得分或者一些
其他类型的标签(分类)或者数值(回归)。

## 函数

> Neural Networks with at least one hidden layer are universal
> approximators(通用 近似器)

经过证明，存在一个连续函数f，以及e为参数且大于0,而一个具有一层隐藏层的神经网络函
数g，对于所有x，都有 `|f - g| < e`
即神经网络近似于任何连续的函数。

一个两层的神经网络为一个通用近似器，其在数学上性质很可爱，但在实际中，其很弱小和
无用。多个隐藏层将会比当个隐藏层的效果更好，同时也适用使用梯度优化器。

## 结构

为一个网状结构，从输入数据，经过隐藏层的映射，输出对应的结果。

# 预加载

```python
import numpy as np
import matplotlib.pyplot as plt

COLOR = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
```

# 生成数据

``` {.python session="py" results="output graphic" file="./images/example-for-nerual-network-945052.png" exports="both"}
N = 100
D = 2
K = 3
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')
learning_rate = 0.00001
reg_lambda = 0.0001
for j in range(K):
    # 确定每一个类的行范围
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
    # C_将两个数组按列进行连接(合并)
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j


fig, ax = plt.subplots(figsize=(9.0, 6.0))
ax.scatter(X[:, 0], X[:, 1],
           c=list(map(lambda x: COLOR[x], y)),
           s=40, cmap=plt.cm.Spectral)
plt.tight_layout(pad=0.0)
```

![](./images/example-for-nerual-network-945052.png)

# 初始化参数

```python
W = 0.01 * np.random.randn(D, K)
B = np.zeros((1, K))
```

# 线性分类器计算得分

将所有输入乘以权重累加后，再加上偏差。计算出每一个样本，对应3个类的得分，直观上，
渴望正确的类获得更高的得分，换句话说，就是正确的类在三个类的占比应该最大。

```python
def f_scores(X, W, B):
    return np.dot(X, W) + B
```

# 计算损失

使用交叉熵(softmax
classifier)。$ L_i = -log(\frac{e^{f_{yi}}}{\sum_j e^{f_j}})
$，假设以下情况，如果仅仅具有一个类时，那预测就是正确的类，那计算出来的损失应该
为0,而 log(1) = 0
这是取log的原因。在得分中，所在比例越少，说明损失越大，但从log
函数的性质看，在(0, 1)区间中，越接近0,越接近负无穷，所以取负号。

这个数据集的损失等于$L = \frac{1}{N} \sum_i L_i + \frac{1}{2}\lambda \sum_k
\sum_i W_{k,i}^{2} $，表示为样本的平均损失加上正规损失。

下面例子中，y代表着正确类别，同时也是下标。这里由于权重矩阵是随机生成的，所以预
测正确的概率应该为1/3,因此损失值大约为 -log(1/3) =
1.09，跟计算出来的一样。

```python
def f_exp_scores(scores, y):
    height = scores.shape[0]
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(height), y])
    return probs, correct_logprobs

def f_loss(correct_logprobs, W, reg_lambda):
    data_loss = np.mean(correct_logprobs)
    regularztion_loss = 1/2 * reg_lambda * np.sum(W*W)
    loss = data_loss + regularztion_loss
    return loss
```

# 计算梯度

根据公式计算得到，可以直接得出损失函数的梯度。

```python
def f_gradient(X, W, probs, y, reg_lambda):
    dscores = probs.copy()

    height = probs.shape[0]
    dscores[range(height), y] -= 1
    dscores /= height

    dW = np.dot(X.T, dscores) + reg_lambda * W
    dB = np.sum(dscores, axis=0, keepdims=True)
    return dW, dB
```

# 更新权重

```python
def f_update(W, B, dW, dB, learning_rate):
    weights = W.copy()
    bias = B.copy()
    weights -= learning_rate * dW
    bias -= learning_rate * dB
    return weights, bias
```

# 迭代更新

```python
def main(X, W, B, y, reg_lambda, learning_rate, iter_num=100, verbose=False):
    for i in range(iter_num):
        scores = f_scores(X, W, B)
        probs, correct_logprobs = f_exp_scores(scores, y)
        loss = f_loss(correct_logprobs, W, reg_lambda)
        dW, dB = f_gradient(X, W, probs, y, reg_lambda)
        W, B = f_update(W, B, dW, dB, learning_rate)
        if verbose:
            print("iter_num: %d, loss: %f" %(i, loss))
    return W, B, loss

res_w, res_b, res_loss = main(X, W, B, y, 1e-3, 1e-0, iter_num=200, verbose=True)
```

得到线性模型的损失函数为0.73多。

# Neural network

从上面的线性分类器中，看到准确率仅仅51%。采用神经网络对数据进行拟合，设定一个两
层的网络，其中第一层网络具有100个节点，而第二层即最后一层具有3个(节点)分类。

```python
h = 100
W = 0.01 * np.random.randn(D, h)
B = np.zeros((1, h))

W2 = 0.01 * np.random.randn(h, K)
B2 = np.zeros((1, K))

learning_rate = 1e-0
reg_lambda = 1e-3
```

## 计算得分

```python
def n_scores(X, W, W2, B, B1):
    hidden_layer_scores = np.dot(X, W) + B # (300, 100)
    # 激励函数 ReLU
    hidden_layer_scores = np.maximum(0, hidden_layer_scores)

    # 输出层
    output_scores = np.dot(hidden_layer_scores, W2) + B2 # (300, 3)
    return hidden_layer_scores, output_scores
```

## 计算损失

损失函数同样是使用上面的softmax。根据反向传播算法。

```python
def n_data_loss(output_scores, y):
    height = y.shape[0]
    exp_scores = np.exp(output_scores)
    # 得分在各类中的占比
    exp_scores_percent = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corr_scores = exp_scores_percent[list(range(height)), y]
    data_loss = np.mean(-np.log(corr_scores))
    return data_loss, exp_scores_percent

def n_regularztion_loss(W, W2, reg_lambda):
    return 0.5 * reg_lambda * (np.sum(W * W) + np.sum(W2 * W2))

# n_loss = n_data_loss(output_scores, y) + n_regularztion_loss(W, W2, reg_lambda)
```

## 计算梯度

对损失函数求导。

```python
def n_gradient(X, W2, exp_scores_percent, y, hidden_layer_scores):
    # 对softmax函数求导部分，前面已经用公式证明，
    height = y.shape[0]
    doutput_scores = exp_scores_percent.copy()
    doutput_scores[list(range(height)), y] -= 1
    doutput_scores /= height

    # output_scores = np.dot(hidden_layer_scores, W2) + B2
    # dw2 = hidden_layer_scores.T * doutput_scores
    dW2 = np.dot(hidden_layer_scores.T, doutput_scores) # (100, 3)
    dB2 = np.sum(doutput_scores, axis=0, keepdims=True) # (1, 3)

    # output_scores = np.dot(hidden_layer_scores, W2) + B2
    # 先计算output_scores对hidden_layer_scores的导数
    dhidden_layer_scores = np.dot(doutput_scores, W2.T) # (300, 100)
    # Relu求导得，仅仅当x大于0，求导得1
    dhidden_layer_scores[hidden_layer_scores <= 0] = 0

    dW = np.dot(X.T, dhidden_layer_scores)
    dB = np.sum(dhidden_layer_scores, axis=0, keepdims=True)
    return dW, dB, dW2, dB2
```

## 更新函数

```python
def n_main(X, y, h, W, B, W2, B2, learning_rate=1e-0, reg_lambda=1e-3, iter_num=500, verbose=False):
    W = W.copy()
    B = B.copy()
    W2 = W2.copy()
    B2 = B2.copy()
    for i in range(iter_num):
        hidden_ls, output_scores = n_scores(X, W, W2, B, B2)
        data_loss, exp_scores_percent = n_data_loss(output_scores, y)
        reg_loss = n_regularztion_loss(W, W2, reg_lambda)
        loss = data_loss + reg_loss

        d_w, d_b, d_w2, d_b2 = n_gradient(X, W2, exp_scores_percent, y, hidden_ls)
        d_w += reg_lambda * W
        d_w2 += reg_lambda * W2

        # update
        W    -= learning_rate * d_w
        B    -= learning_rate * d_b
        W2   -= learning_rate * d_w2
        B2   -= learning_rate * d_b2

        if verbose and i % 100 == 0:
            print("iter: %d, loss: %f" %(i, loss))
    return W, B, W2, B2, loss

nres_W, nres_B, nres_W2, nres_B2, nres_loss = n_main(X, y, h, W, B, W2, B2, iter_num=10000, verbose=True)

# 计算准确率
_, n_s = n_scores(X, nres_W, nres_W2, nres_B, nres_B2)
n_pred = np.argmax(n_s, axis=1)
np.mean(n_pred == y)
```

线性分类器，得到的损失函数值为0.78,而神经网络得到的损失值为0.24，神经网络的准确
率达到98%。

# 决策边界

## 线性函数决策边界

``` {.python session="py" results="output graphic" file="./images/example-for-nerual-network-480812.png"}
step = 0.02
xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1

# 假设  xx，yy都为 (196, 191)
xx, yy = np.meshgrid(np.arange(xmin, xmax, step),
                     np.arange(ymin, ymax, step))

# 将矩阵拉平后，在合并成(196x191, 2)
# (196x191, 2) 再与W权重矩阵相乘W(2, 3)，得到一个
# (196x191, 3)其中每一行代表一个样本3个类各自的得分。
# Z相等与后面的得分，只不过这里不是用X，而是用xx，yy
# 因此可以使用上面的计算得分的函数进行计算
# Z = f_scores(np.c_[xx.ravel(), yy.ravel()], res_w, res_b)
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], res_w) + res_b

# 从中选择最大的概率的类。
Z = np.argmax(Z, axis=1)        # (196x191, 1)
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(9.0, 6.0))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdPu, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdPu, edgecolors='black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.tight_layout(pad=0.0)
```

## 神经网络决策边界

``` {.python session="py" results="output graphic" file="./images/example-for-nerual-network-959414.png"}
# 计算在xx，yy下的得分
_, Z = n_scores(np.c_[xx.ravel(), yy.ravel()], nres_W, nres_W2, nres_B, nres_B2)
# 从中选择最大的概率的类。
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(9.0, 6.0))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdPu, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdPu, edgecolors='black')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.tight_layout(pad=0.0)
```

# 参考

[CS231n](https://cs231n.github.io/neural-networks-case-study/)
