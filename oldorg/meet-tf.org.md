TensorFlow 为Google开源的机器学习框架。其整个架构分为三个部分:

-   数据预处理
-   构建模型
-   训练模型以及使用模型预测

被称为\"TensorFlow\"的原因在于：其输入数据为多维数组(tensor)，而各种操作函数组成各
个节点，当输入数据用各种操作函数执行时，组成一个流程图(flowchart)。

TensorFlow适用于多个平台，甚至手机也可以。其具有两大组件：

-   Tensor: 用来创建数据、指定数据的类型、形状等，还用来创建操作节点(op
    node)。
-   Graph: 用来组合上面的Tensor，利用各个节点构建一个网络(CNN,
    RNN等等)。

TensorFlow中还提供来加载数据的API，用来批量加载大数据集，不同全部数据都加载到可
限的内存中去。其方法是通过将数据集占位符转变成一个可迭代的生成器，从生成器中获取
每次的小批量数据。

TensorFlow给我的初步印象就是：先通过占位符将各个数据变量、操作函数去构建一个模型
流程图，然后在训练时才把数据传进去训练。

TensorFlow中的Tensor具有三个属性：

-   名字：一个独一的标签
-   维度：形状
-   数据类型：tensorflow将会自动的判断所对应的类型。

当没有制定名字的时候，会默认基于 `"const_x"`{.verbatim} 命名。使用
`tf.constant`{.verbatim} 创建一个常 量。

``` {.python session="py" results="output" exports="both"}
import tensorflow as tf

type_float = tf.constant(3.141519, tf.float32)
type_int = tf.cast(type_float, dtype=tf.int32, name="float_to_int")
print(type_float)
print(type_int)
```

``` {.example}
Tensor("Const:0", shape=(), dtype=float32)
Tensor("float_to_int:0", shape=(), dtype=int32)
```

TensorFlow中内置许多基本操作函数，像绝对值、指数、对数、开方等。值得注意的是，这
些操作函数也并不是返回一个结果，而是创建一个占位符，表示相关元素需要进行该操作。

``` {.python session="py" results="output" exports="both"}
tensor_a = tf.constant([[1, 2]], dtype=tf.float16)
tensor_b = tf.constant([[3, 4]], dtype=tf.float16)
tensor_add = tf.add(tensor_a, tensor_b)
print(tensor_add)
```

``` {.example}
Tensor("Add:0", shape=(1, 2), dtype=float16)
```

我们需要用一个变量实时更新结果，可以使用=tf.get~variable~=
创建变量或者获取变量。

Note: TensorFlow中默认是不允许重复定义同一 `name`{.verbatim} 的变量。

``` {.python session="py" results="output" exports="both"}
var_int_1 = tf.get_variable("tensor_var_1", shape=[1, 2], dtype=tf.int32, initializer=tf.zeros_initializer)
print(var_int_1.shape)
```

``` {.example}
(1, 2)
```

使用 `tf.placeholder`{.verbatim} 与 `tf.feed_ditc`{.verbatim}
来定义占位符和提供占位符所需的数据。提供 `name`{.verbatim}
的一个好处就是会在输出的结构图(或者流程图flowchart)中将name显示为名称。

``` {.python session="py" results="output" exports="both"}
data_place_holder_1 = tf.placeholder(tf.int16, shape=[1, 2], name="data_place_holder_1")
print(data_place_holder_1)
```

``` {.example}
Tensor("data_place_holder_1:0", shape=(1, 2), dtype=int16)
```

`tf.session`{.verbatim}
当整个模型的网络构建完后，TensorFlow将会创建一个会话对网络按照各个
节点的操作函数进行求值。

``` {.python session="py" results="output" exports="both"}
x = tf.constant([2])
y = tf.constant([4])
multiply = tf.multiply(x, y)
with tf.Session() as tfs:
    result_1 = tfs.run(multiply)
    print(result_1)
```

``` {.example}
[8]
```

TensorFlow如此流行还有一个原因，其提供 \`TensorBoard\'
可视化工具，对调试、优化都 有很大的帮助。TensorFlow
中将计算等操作视为底层 API, 而将一些集成的机器学习算法
视为高层API(估计器estimator)。
