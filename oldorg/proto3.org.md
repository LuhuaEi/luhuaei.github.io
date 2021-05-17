[proto3](https://developers.google.com/protocol-buffers/docs/proto3)

# style guide

1.  license
2.  overview
3.  syntax
4.  package
5.  File options
6.  other

# 声明语法

    syntax = "proto3";

# 注意点:

1.  信息队列编码, 1-15为单字节，16-2047为双字节，编码范围为1-2^29^.
    在这期间 19000:19999为预留的区间，不推荐使用。

# 导入其他proto文件

1.  如果具有多个文件，比如A, B, C
2.  加载proto路径，通过 -I/--proto~path~ 指定目录

```{=html}
<!-- -->
```
    import "foo/foo.proto";

    // A文件(A.proto)
    import "Apple/apple.proto";

    // B文件(B.proto)
    import "Boy/boy.proto";
    import public "A.proto";

    // C文件(C.proto), C文件可以调用: A, 但不可以调用Boy/boy
    import "B.proto";

# 信息(message)

1.  单数(singular), 接收信息范围: 0, 1， 在proto3中为默认的选择。
2.  重复(repeated), 接收任何数量的信息
3.  多个信息定义可以处于一个proto文件中
4.  使用reserved保留关键词或者信息编码， 如2, 15, 9 to 11, \"foor\",
    \"bar\"
5.  在message中可以使用repeated重复使用
6.  支持在结构体中定义其他的结构体(nested type)

```{=html}
<!-- -->
```
    message Foo {
      repeated Bar bar = 1;
      message _Foo {
        string otherFoo = 1;
      }
      repeated _Foo foo = 2;
    }
    message Bar {
      string title = 1;
      string content = 2;
      repeated string name = 3;
    }

1.  使用大写的驼峰式名字。结构体类型使用下划线的名字。
2.  对重复的字段使用复数名称(repeated)

## 信息更新

1.  当需要更新一个领域的结构体时，不需要重新设计，而可以在已有的领域上加入新的代
    码。
2.  新的代码可以解析和兼容老的代码，而老的二进制文件对新加入不能识别的地方进行忽
    略。
3.  可以对老的message进行移除，但更加推荐使用OBSOLUTE~前缀进行标注或者使用~
    reserved进行预留，以便后用。
4.  int32, uint32, int64, uint64, bool
    之间可以相互变换，而不用考虑各个之间的兼容
    性。（从int64转为int32将会被截断）。其他类型类似。
5.  sint32, sint64相互兼容，但是不兼容其他类型。
6.  string, bytes 是相互兼容的。
7.  fixed32, sfixed32, fixed64, sfixed64 相互兼容。
8.  enum(枚举值)兼容int32, uint32, int64, uint64,
    不能反序列化的信息，都会保留下
    来，而int类型总是保留（保留下来是值留给客户端的程序进行解码）

# 默认值

字符串 空字符串 字节 空字节 布尔值 false 数值型 0 枚举型
第一个元素且必须为0

# 枚举型

1.  枚举第一个元素必须对应为0
2.  在将allow~alias打开后~，可以给不同的key赋予相同的value
3.  枚举值推荐使用32-bit的数值，
4.  在反序列化中，不能识别的枚举值将会保留下来给后面的程序解析。

```{=html}
<!-- -->
```
    enum corpus {
      option allow_alias = true;
      foo = 0;
      bar = 1;
      bar2 = 1;
    }

1.  使用大写的驼峰式作为名称，在结构体中使用全大写加下划线作为类型名。

# 预留值

保留特定的值，以便后面用到(不能将数值型跟字符串型混合在同一条语句上)

    enum corpus {
      reserved 0, 1, 5 to 10, 1000 to max;
      reserved "Foo", "bar";
    }

# 无知领域(unknown fields)

proto3在刚开始的时候，是将不能识别的语法（区域）都抛弃，但是在3.5后，对不知的区
域信息将保留下来作为序列化输出的一部分。

# Any类型

使用Any类型需要从google/protobuf/any.proto中导入。Any类型可以让你加入一个没有定
义的类型。(不同的语言对Any类型的解码的函数也不同)

    import "google/protobuf/any.proto";
    message Foo {
      string message = 1;
      repeated google.protobuf.Any details = 2;
    }

# oneof(之一类型)

1.  主要是可以在多个message结构中使用同一个类型。跟正规的结构类型类似，只是oneof
    在多个结构之间通过分享内存共用。
2.  使用case() 或者 WithOneof() 来判断是否属于oneof类型
3.  oneof结构中，可以添加任何类型
4.  可以在任何结构体中，添加oneof类型，除了repeated类型。

```{=html}
<!-- -->
```
    message foo {
      oneof test_foo {
        string name = 4;
      }
    }

1.  如果在多处设置oneof,
    将会仅仅在最后一个设置保留值，而其他的地方将会被自动清理。(?)
2.  如果一个解析器对含有oneof的信息进行解析，将会最后一个遇到的oneof领域才能被用
    来解析。
3.  oneof 不能被repeated(4)
4.  Reflection API 是通过 oneof 的原理
5.  默认值将会被序列化。
6.  小心删除oneof类型数据，即使一个oneof类型返回None/NOT~SET~，这个只能表示这个值
    没有被使用或者已经被使用过。

# Map(关联表)

1.  key类型不能是枚举类型或浮点数，可以是整数和字符串
2.  value类型可以是任何类型，除了map类型
3.  下面的例子中，将表示在Project中，所有的类型都是为String类型。

```{=html}
<!-- -->
```
    map<string, Project> projects = 3;

1.  map类型不能repeated
2.  在汇编中，map是无序的
3.  在生成的proto文件，map会被排序成按key，或者数值排序。
4.  从汇编中解析时，如果存在重复的key将会使用最后一个key，而如果从proto文件中解析
    时，存在重复key就会报错。
5.  如果一个map中有空的key，在一些语言中，会使用用默认值进行序列化，而另一些语言
    中将不会进行序列化。
6.  map 兼容(不兼容的可以使用以下的方法进行模拟， 两者在编译后是一样的)

```{=html}
<!-- -->
```
    message Entry {
      key_type key = 1;
      value_type value = 2;
    }
    repeated Entry foo = 1;

# Package (namespace)

1.  可以使用package来避免两个文件之间的名字冲突,
    不同的语音使用的实现方法也不一样。

不同的语言使用的实现方法也不一样。

1.  包名应该小写

# Service 接口

1.  使用在RPC服务

```{=html}
<!-- -->
```
    service SearchRequest () {
      rpc Search (SearchRequest) returns (SearchResponse);
    }

1.  使用大写的驼峰式（结构名和类型名）

# JSON Map

1.  如果值为null，在编码成proto buffer时，就会替换成合适的默认值。
2.  如果在proto buffer 中已经具有默认值时，将会忽略编码。

  ------------------------ --------------- -------------------------------------------------
  proto3                   JSON            JSON example
  message                  object          {\"fooBar\": v,\"g\": null,...}
  enum                     string          \"FOO~BAR~\"
  map\<K,V\>               object          {\"k\": v, ...}
  repeated V               array           \[v, ...\]
  bool                     true, false     true, false
  string                   string          \"Hello World!\"
  bytes                    base64 string   \"YWJjMTIzIT8kKiYoKSctPUB+\"
  int32, fixed32, uint32   number          1, -10, 0
  int64, fixed64, uint64   string          \"1\", \"-10\"
  float, double            number          1.1, -10.0, 0, \"NaN\", \"Infinity\"
  Any                      object          {\"@type\": \"url\", \"f\": v, ... }
  Timestamp                string          \"1972-01-01T10:00:20.021Z\"
  Duration                 string          \"1.000340012s\", \"1s\"
  Struct                   object          { ... }
  Wrapper types            various types   2, \"2\", \"foo\", true, \"true\", null, 0, ...
  FieldMask                string          \"f.fooBar,h\"
  ListValue                array           \[foo, bar, ...\]
  Value                    value           
  NullValue                null            
  Empty                    object          {}
  ------------------------ --------------- -------------------------------------------------

1.  通过添加选项来实现忽略z无知结构类型(unknown fields)

# Options

1.  一些选项是应用于文件的，应该写在文件的头部
2.  一些选项是应用于结构体中，应该写在类型定义的前面。
3.  导入go packge

```{=html}
<!-- -->
```
    option go_package = "../foo/bar";

# 类型表

  ------------- ---------- ------------------ ----------
  .proto Type   C++ Type   Python Type\[2\]   Go Type
  double        double     float              float64
  float         float      float              float32
  int32         int32      int                int32
  int64         int64      int/long\[3\]      int64
  uint32        uint32     int/long\[3\]      uint32
  uint64        uint64     int/long\[3\]      uint64
  sint32        int32      int                int32
  sint64        int64      int/long\[3\]      int64
  fixed32       uint32     int/long\[3\]      uint32
  fixed64       uint64     int/long\[3\]      uint64
  sfixed32      int32      int                int32
  sfixed64      int64      int/long\[3\]      int64
  bool          bool       bool               bool
  string        string     str/unicode\[4\]   string
  bytes         string     str                \[\]byte
  ------------- ---------- ------------------ ----------
