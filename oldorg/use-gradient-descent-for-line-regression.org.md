**梯度下降**
为一种优化方法，其原理在于：通过对损失函数(凸函数)进行求导，寻找极小
值所在的方向，沿着该方向前进，直到其达到极小值(可能是全局极小值、也有可能为局部
极小值)。

在一个凸函数中，假设其仅有一个极小值点，当某点位于极小值点左边时，模型需要朝右边
前进才是正确的方向，当位于右边时，模型需要往左边前进，才可能找到极小值点。

在数学上， **梯度**
表示成一个多变量函数的全导数(即所有可能的偏导数的组合)。具有一
个参数的函数，在二维空间上表示成一条线，而具有两个参数的函数，在三维空间上表示成
一个平面，而梯度下降就是要找到其中的极小值，对于一个凸曲线来说，极小值处于导数为
0的地方，而对于一个凸面来说，极小值就是处于所有曲线的极小值交点处。

# 优化器

梯度下降在寻找最小值的过程中，就是一个优化过程。而关于 **优化器**:
优化器用于根据
损失函数寻找最优的权重值，使得损失函数值最小。我们不可以寻找到最好的权重值在一开
始，而是要通过交互改进的方法进行寻找，即先设定一个随机的权重值，通过每一次对权重
进行修正。

> Our strategy will be to start with random weights and iteratively
> refine them over time to get lower loss

## 随机搜索

先随机生成一个权重，对给予的权重值求值损失函数，那当前的损失函数与目前最优的损失
函数相比，如果更优，则保留该权重，否则重复生成一个随机新的权重值，重复比较。

## 随机局部搜索

先随机生成一个权重，每一次都是通过微调权重，如果损失函数更小，则选择为当前最优的
权重，否则，继续对当前的权重进行微调。微调可以为那当前的权重值W，加上一个aw，其
中a代表微调的系数(步伐大小)，w代表一个新的随机权重值。

这个方法主要的要点在于，如果随机生成的w权重值与降低损失的方向相同，则更新当前的
权重，如果不同，则忽略。

## 跟随梯度搜索

而跟随梯度搜索为通过梯度可以知道最小损失所在的方向，因此，不需要寻找方向，只需要
更新权重值，令损失达到最小。

实现一个梯度下降函数。梯度是一个矢量，具有大小与方向两个特征。梯度始终会指向损失
函数增长最大的方向(通过寻找损失函数中的最大绝对值)而梯度下降就是沿着负梯度的方向
前进(以便尽快降低损失函数)。而下一个前进的点就是当前的点加上梯度的大小，依次重复，
直到极小值(可能为局部极小值，也有可能为全局极小值)。

在数学上，梯度表示成一个多变量函数的全导数(即所有可能的偏导数的组合)。具有一个参
数的函数，在二维空间上表示成一条线，而具有两个参数的函数，在三维空间上表示成一
个平面，而梯度下降就是要找到其中的极小值，对于一个凸曲线来说，极小值处于导数为0
的地方，而对于一个凸面来说，极小值就是处于所有曲线的极小值交点处。

就好比一个人处于一个盆地的上方，先到达盆地的最低点，对与整个盆地中，除了极小值点
外，其他所有的点所对应的偏导数都是大于0的，即指向损失函数增大的方向，而梯度下降
就是沿着负梯度下降。

### 数值法梯度(numerical gradient)

特点为：速度慢，但是容易操作。用泰勒表达式可以验证用一个h的误差项比2h的误差项大。
所以推荐使用2h求导。$g = \frac{f(x + h) - f(x - h)}{2h}$

``` {.python session="py" results="output silent" exports="both"}
def eval_gradient_numerical(f, x, verbose=True, h=0.00001):
      '''
      Inputs:
      ------------------------------------------------------------
      - f: (accept one paramter) loss function.
      - x: (data set) The point to evaluate the gardient at
      Outputs:
      ------------------------------------------------------------
      - grad: (shape same with x) The gradient on each point of x.
      '''
      # 生成一个与输入x相同维度的0数组
      grad = np.zeros_like(x)
      # 多维迭代，这里对x的每一个维度都进行迭代
      # 先对最后的维度进行迭代，当完成后，再对前一个维度进行迭代
      # 迭代完成后，使用finished判断
      it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
      while not it.finished:
          ix = it.multi_index
          old_val = x[ix]

          x[ix] = old_val + h
          fxph = f(x)

          x[ix] = old_val - h
          fxmh = f(x)

          x[ix] = old_val

          # 计算斜率，这里使用斜率的原始定义计算，通过微小的面积与h宽度。
          grad[ix] = (fxph - fxmh) / (2 * h)

          if verbose:
              print(ix, grad[ix])
          it.iternext()
      return grad

def eval_gradient_array(f, x, df, h=1e-5):
    '''
    Inputs:
    ------------------------------------------------------------
    - f: loss function
    - x: input data set
    - df: the loss function output gradient array

    Outputs:
    ------------------------------------------------------------
    - grad: the grad is the gradient array.
    '''
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]

        x[ix] = old_val + h
        pos = f(x).copy()

        x[ix] = old_val - h
        neg = f(x).copy()

        x[ix] = old_val
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
```

### 解析法梯度(analytic gradient)

特点为：数据快，但需要操作精准(易于出错)。利用微积分的原理求解梯度，先把表达式用
积分计算出来，再在代码中实现(求导数)。在现实中，一般采用分析梯度，然后对结果进行检验，即
与数字梯度进行比较。可能一个函数很难或者无法计算导数。

### 反向传播梯度(backward pass)

由于梯度其实就是斜率，也就是函数的导数，通过解析法可以直接求出最后的值，而反向传
播是一步一步向后求导数，只要不是到最后一步，即只要不是到达输入数据层，都被视为一
个函数，再通过链式法则求各个节点的导数。

在反向传播过程中，最常见的节点类型具有加、乘、取最大值。当遇到 **加号**
时，无论前
向传播中是什么值，而在方向传播过程中，该节点输出值保持与输入值不变；当遇到
**乘 号** 时，如 `x*y` 对x求导就是y，对y求导就是x；当遇到
**最大值符号** 时，最大值符号 函数可以理解成一个分段函数
`max(x, y)` 当 `x > y`
时，就是对x进行求导，否则就对y
求导，所以需要考虑该节点的前向传播中所有的输入值。

### 梯度审核

梯度检验：使用相对误差来比较两个梯度的差异，这是因为如果使用绝对误差，0.00001在目
标值0.00001和1中是大小程度不相同的。几条检验规则：

  relative error   situation
  ---------------- -----------------------------------------------------------------------
  \>1e-2           梯度错误
  (1e-2, 1e-4)     怀疑(很有可能某个计算出错)
  \<1e-4           对于一些复杂的目标函数可能很好，但对于tanh、softmax这些来说，还是太高
  \<1e-7           表现良好

需要注意的是，在计算梯度中，使用双精度类型将会更好。而Kinks是导致梯度验证不准确
的一个因素。Kinks与目标函数的不可微分有关、kinks可以通过使用更少的数据集进行验证，
更小的数据集同样令的程序更加高效。

### 权重更新

当利用梯度下降的方法求的每一个位置的梯度后，利用这些梯度对已有的权重进行更新，这
是一个学习的过程，不同的学习速率具有不同大小的前进步伐。 `W_new = W -
learning_rate * grad`
注意到这里对权重更新是那当前的权重减去对应的梯度，这是因为
斜率的正负表示函数的单调性，如果斜率为正，那随着自变量的增大，因变量也会增大；如
果斜率为负，那随着自变量的增大，因变量减少；而在优化过程中，权重值(W)作为自变量，
损失值作为因变量。而优化器的目的在于是损失达到最小(达到极小值)，故沿着负梯度的方
向前进。

梯度只是告诉正确的前进方向，而没有告知前进的步长，步长太短，计算量大，速度慢；步
伐太长，容易超过最低点，甚至还会造成不收敛的情况发生(处于极值点附近动荡)。

# 反向传播

方向传播的主要原理在于使用链式法则。通过前向传播的步骤，一步一步反推。

## 加载数据

``` {.python session="py" results="output silent" exports="both"}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('ggplot')
DATA = np.array(np.genfromtxt("data/gd-line-regression.csv", delimiter=','))
```

## 数据预览

``` {.python session="py" results="output graphic" file="./images/use-gradient-descent-for-line-regression-945387.png" exports="both"}
x = DATA[:, 0]
y = DATA[:, 1]
plt.figure(figsize=(9.0, 6.0))
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout(pad=0.0)
```

![](./images/use-gradient-descent-for-line-regression-945387.png)

## 梯度下降函数

``` {.python session="py" results="output silent" exports="both"}
class LinearRegression():
    def __init__(self):
        self.weights_list = []

    def model(self, x_train, y_train):
        '''
        Inputs:
        ------------------------------------------------------------
        - x_train: (N, D)
        - y_train: (N, 1)
        '''
        self.x_train = x_train
        self.y_train = y_train

    def train(self, weights, learning_rate=0.0001, iter_count=1000):
        '''
        Inputs:
        ------------------------------------------------------------
        - weights: (D, 1)
        - learning_rate: (float)
        - iter_count: (integer)
        Outputs:
        ------------------------------------------------------------
        update the weights on self weights.
        '''
        for i in range(0, iter_count):
            gw = self.gradient_descent(weights)
            weights -= learning_rate * gw
            # if don't copy will lead all values will the last update value.
            self.weights_list.append(weights.copy())
        self.weights = weights


    def gradient_descent(self, w):
        '''
        use forward and backward pass computer the gradient

        Inputs:
        ------------------------------------------------------------
        - w: (D, 1) weights
        Outputs:
        ------------------------------------------------------------
        - dw: (D, 1) gradient weights
        '''
        # forward pass
        w_mul_x = self.x_train.dot(w)                           # 1 (N, 1)
        w_mul_x_sub_y = w_mul_x - self.y_train                  # 2 (N, 1)
        w_mul_x_sub_y_square = np.square(w_mul_x_sub_y)         # 3 (N, 1)
        sum_w_mul_x_sub_y_square = np.sum(w_mul_x_sub_y_square) # 4 (1, 1)
        mse = sum_w_mul_x_sub_y_square / self.x_train.shape[0]  # 5 (1, 1)

        # backward pass
        dsum_w_mul_x_sub_y_square = 1 / self.x_train.shape[0] # 5
        # 4 equal 1 * dsum_w_mul_x_sub_y_square * self.x_train.shape[0]
        dw_mul_x_sub_y_square = 1
        dw_mul_x_sub_y = 2 * w_mul_x_sub_y * dw_mul_x_sub_y_square # 3 (N, 1)
        dw_mul_x = dw_mul_x_sub_y                                  # 2 (N, 1)
        dw = self.x_train.T.dot(dw_mul_x)                          # 1 (D, 1)
        return dw

    def MSE(self):
        '''computer the mean square error'''
        y_pred = np.dot(self.x_train, self.weights)
        return np.mean(np.square(self.y_train - y_pred))

    def plot(self, w):
        plt.figure(figsize=(9.0, 6.0))
        plt.plot(self.x_train[:, 1], self.y_train, 'bo')
        plt.plot(self.x_train[:, 1], np.dot(self.x_train, w), 'r-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout(pad=0.0)
        plt.show()
```

## 求解

这里已经将截距项合并到x中以及weights中。

``` {.python session="py" results="output graphic" file="./images/use-gradient-descent-for-line-regression-134697.png" exports="both"}
# 在数据前面添加一列，用来计算截距项
xt = np.c_[np.ones((x.shape[0])), x]
yt = y.reshape(y.shape[0], -1)
# 两个系数，一个截距项，一个系数
weights = np.zeros((2, 1))

linear = LinearRegression()
linear.model(xt, yt)
linear.train(weights.copy(), learning_rate=0.000001, iter_count=10)
linear.plot(linear.weights)
```

![](./images/use-gradient-descent-for-line-regression-134697.png)

## 优化过程

``` {.python session="py" results="output silent" exports="both"}
fig, ax = plt.subplots(figsize=(9.0, 6.0))
ax.scatter(x, y)
line, = ax.plot(x, np.dot(xt, weights), 'r-', lw=3)

def update(i):
    y_pred = np.dot(xt, linear.weights_list[i])
    line.set_ydata(y_pred)
    return line,

anim = FuncAnimation(fig, update, frames=list(range(10)), interval=500)
anim.save('./images/update-line-for-gradient-descent.gif', fps=60, writer='imagemagick')
```

![](./images/update-line-for-gradient-descent.gif)
