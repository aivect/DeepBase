---
layout: article
title: "Python 基础语法"
key: learning-01-python
permalink: /learning/01-python.html
sidebar:
  nav: learning
aside:
  toc: true
comment: true
mathjax: true
show_edit_on_github: false
show_subscribe: false
sharing: false
---



# 1 基础语法

## 1.1 python 解释性、动态且强类型语言

python**解释性**（读取一行，执行一行），面向对象的语言，动态类型语言。

cmd运行：python + 文件地址

| 类型检查 | 确保一个表达式中的变量类型是合法的 |
| ---------------- | ------------------------------------------------------------ |
| **静态类型语言** | 类型检查发生在编译阶段(compile time)，需要显示声明类型，如Java |
| **动态类型语言** | 如果类型检查发生在运行阶段(run time)，不需要显示声明类型，如Python |
| 强类型语言 | 有更强的类型检查机制，表达式计算中会做严格的类型检查 |
| 弱类型语言 | 允许各种变量类型间做一些运算。 |

```python
data = 5
data = data + 'xiaoming'
# python执行报错，因为强类型
```

```javascript
var data = 5
data = data + 'xiaoming' \
//javascript：string和int可以结合,程序正常执行
```

Python是一门动态的(dynamic)且强类型(strong)语言

Python是动态类型语言，变量不需要显式声明类型。根据变量引用的对象, Python解释器自动确定数据类型。
Python是强类型语言
每个对象都有数据类型，只支持该类型支持的操作。

## 1.2 Python程序的构成

1、python程序有模块组成，一个模块对应Python源文件，一般后缀名：.py

2、模块有语句组成。

python使用缩进而不是{}程序块。

## 1.3 python一切皆对象

Python中，一切皆对象。每个对象由:**标识( identity )、类型( type)、value (值)**
组成。

1.**标识**用于唯一标识对象，通常对应于对象在计算机内存中的地址。使用内置函数id(obj)
可返回对象obj的标识。

2.**类型**用于表示对象存储的”数据”的类型。类型可以限制对象的取值范围以及可执行的
操作。可以使用type(obj)获得对象的所属类型。

3.**值**表示对象所存储的数据的信息。使用print(obj)可以直接打印出值。

对象的本质就是: 一个内存块，拥有特定的值，持特定类型的相关操作。

```python
#4是一个对象，a是一个变量
a = 4
if (id(4) id(a) and type(4) type(a)):
    print(True)
else:
    print(False)

    '''
output:
True
'''
```

<img src="{{ '/assets/learning/01_python/1690024762103.png' | relative_url }}" alt="1690024762103" style="zoom: 80%;" />

python的一切变量都是对象，变量的存储采用了引用语义的方式，存储的只是一个变量的值所在的内存地址，而不是这个变量的本身。

**变量位于栈内存，对象为与堆内存**。

**引用语义**：在python中，变量保存的是对象(值)的引用，我们称为引用语义。采用这种方式，变量所需的存储空间大小一致，因为变量只是保存了一个引用。也被称为对象语义和指针语义。

**值语义**：有些语言采用的不是这种方式，它们把变量的值直接保存在变量的存储区里，这种方式被我们称为值语义，例如C语言，采用这种存储方式，每一个变量在内存中所占的空间就要根据变量实际的大小而定，无法固定下来。

## 1.4 字符串格式

1、单行字符串：' ' 或 " "。不区分双单引号

2、多行字符串：

"""

"""

3、f动态修改字符串: **f**"……{参数名}……"

4、区分大小写

5、\ 行连接符

## 1.5 注释

```python
#单行注释
'''多行注释'''、"""多行注释"""
```

## 1.6 变量

1.必须是数字、字母或下划线组成。

2.不能是数字开头、纯数字、python关键字

3.不能用中文

4.推荐用下划线或驼峰命名（name_of_teacher,NameOfTeacher)

**变量的声明和赋值用于将一个变量绑定到一个对象上**，格式如下:

**变量名=表达式**

最简单的表达式就是**字面量**。比如: a=123 。运行过程中， 解释器**先运行右边**的表达式，**生成**一个代表表达式运算结果的**对象**;然后，将这个**对象地址赋值给左边的变量**。

### 1.6.1 del删除变量

格式：del 变量

### 1.6.2 垃圾回收机制

删除变量后，如果对象没有变量引用。就会被垃圾回收器回收，清空内存空间。

### 1.6.3 赋值

**链式赋值**。链式赋值用于同一个对象赋值给多个变量。

```python
x=y=123
```

**系列解包赋值**。系列类型数据赋值给对应相同个数的变量，个数需保持一致。**实现变量互换**。

```python
# 解包
d,e = 4,5 #或者 d,e = (4,5)
print(d,e)
# 解包实现变量互换。
a,b = 1,2
print(a,b)
a,b = b,a
'''
output:
4 5
1 2
'''
```

## 1.7 常量

常量名用大写

## 1.8 数据类型

1.数字：整数、小数。

| 运算符 | 描述 |
| ------ | ----------------------------------------- |
| + | |
| - | |
| * | |
| / | 浮点数除法 |
| // | 整数除法 |
| % | 模。可用divmod(arg1,arg2)得到商和余数元组 |
| ** | 幂 |

2.文字：

展示：字符串（str)"", ''单引号，双引号都可；''' ''''（多行）

操作： **+** 字符串拼接符 ；一个字符串只能**乘以一个数字**，表字符串重复次数

```python
a = "adsa"*2
print(a)
# adsaadsa
```

3.布尔：Ture,False 表条件判断

**type(变量) 查看变量数据类型**

## 1.9 用户交互

输入：var = input("请输入：")

输出：print(*args, end)

强制数据类型转换：var_b = int(a)

## 1.10 条件判断

1.if	条件：

2.if else

```python
a = str()
if a is "":
	print("空")
else:
	print("非空")
```

3.if a `is` None

4.if elif 

```
if 条件1：
	代码1
elif 条件2：
	代码2
elif 条件3：
	……
else：
```

5.三元条件运算符

```python
# 三元条件运算符
s = input("请输入一个数字：")
print("s<10" if int(s)<10 else "s>10")
```

## 1.11 循环语句

**1**.while 条件：(**while True: ……死循环 **用的多)

代码

----break和continue

**2**.for 变量 in 可迭代的东西(可为字符串、range（1,10）)：

代码

```python
a = "dfgdgfg"
	for p in a: #字符串循环遍历字符串a
print(p)
```

---for循环欲计数 ：

range(n):从0到n，不包含n

range(m,n)：从m到n，不含n

range(m,n,q):从m到n，不含n，每次间隔q

```python
for i in range(3,10,2):
	print(i)
#3,5,7,9
list(range(3,10,2))
```

3.for循环解包+列表推导式

```python
c = [ temp.rstrip()+"#"+str(index) for index, temp in enumerate(a)]

```

```python
idx_to_char = list(set([word for review in reviews
 		for word in review]))
```

内层的列表推导式使用了两个嵌套的for循环，用于遍历reviews列表中的每一个review，以及每个review中的每一个word。这段代码的作用是从多个评论中提取出所有的单词，并返回一个去重后的单词列表。

## 1.12 时间

计算机中时间的表示是从“1970年1月1日00:00:00” 开始，以毫秒(1/1000秒) 进
行计算。我们也把1970年这个时刻成为“unix 时间点”。
这样，我们就把时间全部用数字来表示了。

![1690026131493]({{ '/assets/learning/01_python/1690026131493.png' | relative_url }})

python中可以通过time.time()获得当前时刻，返回的值是以秒为单位，带微秒(1/1000
毫秒)精度的浮点值。例如: 1530167364.8566.

## 1.13 pass

关键字：pass

防止报错，先过，保证代码完整性

```
if a>10:
	pass
else:
```

## 1.14 运算符

1、算数运算：+ - * / % //(整除) **(次方)

2、比较运算：> < >= <= !=

3、赋值运算：= += -= *= ……

4、逻辑运算：and or not 运算顺序：括号>not>and>or

and和or同时出现的时候，尽量写括号

5、成员运算：in not in

in 判断xxx是否在xxx中

# 2 数据类型

## 2.1 Sequence

是一个抽象基类，它定义了序列的共同接口，包括索引、切片、追加和扩展等操作。这意味着，任何类只要实现了这些方法，就可以被视为一个序列。

list、tuple、set、dict等都继承了Sequence的接口，因此它们都可以进行索引、切片、追加和扩展等操作。不过，list和tuple在实现这些方法时，可能会采用不同的实现方式，以满足各自的特性需求。

## 2.2 基础数据类型讲解

int 、float、bool；**str**、**list**、tuple、set、**dict**、bytes、运算符、**文件操作**。**在python中0，空字符串为False,非零，字符串表True**,表示*`空`*都为False

```python
a=1
b=bool((a))
a = "ddfd"
b = bool(a)
print(type(a))
```

## 2.3 字符串(str)

**字符串的操作对原字符串是不会发生改变的。可通过另一字符串变量接收对原字符串的操作**。python不支持单字符类型，单字符仍是作为一个字符串类型使用的。

**1、字符串格式化问题**:%s字符串占位符，%d整数占位符，%f,占位小数

age = **int**(input("请输入:"))

s1= "我叫%s，今年%s岁" % （name,age) #一个变量可不加括号

s2 = "我叫{}，今年{}岁" .format(name,age)

s3 = f"我叫{name}，今年{age}岁" (python 3.6后才有)

**2、索引和切片**

***2.1、索引***：按照位置提取字符串中某一个元素。索引超过范围报错

s = "我家住在黄土高坡"

```python
print(s[0])
print(s[-2])
```

***2.2、切片***：从一个字符串提取一部分内容，从

print(s[start:end]) #从索引位置start开始，end为结束，取不到是s[end]

print(s[start:end :step])#表示从end到start每隔step个打出一个。

print(s[:]) #打出s整个字符串

print(s[-3:0])#打出：土高坡

print(s[0:-3])#无结果。

#可以给切片添加步长，控制方向，默认是从左到右的顺序

print(s[start:end :-1]) # 表示从start到end-1的逆序。步数非负，则为正序。

| 元素内容 | 1 | 2 | 3 | 4 | 5 | 6 |
| :------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 正数索引 | 0 | 1 | 2 | 3 | 4 | 5 |
| 负数索引 | -6 | -5 | -4 | -3 | -2 | -1 |

切片超出索引范围：当start或stop超出上文提到的有效索引范围时，切片操作不会抛出异常，而是进行截断。 start和stop都是可以缺省的，在缺省的情况下，Python的行为是尽可能取最大区间（包含头、尾） 。

**3.替换、切割、合并**

a.upper() #实现对字符串a的大写化

小写：

```python
.lower()
```

**3.1、strip()** #去掉字符串中的前后左右两端的**空白符(**空格，\t缩进，\n 换行符)

```python
b = input("请输入:").strip()
```

3.2、**strip(参数str)** 。会删除收尾指定参数。 传的参数不管是“abc”还是abc的其他排列形式，**函数都会将参数拆分成一个个的字符**，然后循环把头尾的这些字符去掉。

```python
a="aabcacb1111acbba"
print(a.strip("abc"))
print(a.strip("bac"))
#输出：1111。
```

**lstrip()和rstrip()** 分别去掉左边头部，一个是去掉右边尾部

**3.2、replace(old, new) #字符串替换**

```python
a = "hello, i am a good man"
a = a.replace(" ","")
print(a)
```

**3.3、str.split(sep=None, maxsplit=-1)**

- `sep`：分隔符，默认为所有空白字符（空格、换行符 `\n`、制表符 `\t` 等）。
- `maxsplit`：最大分割次数，默认 `-1`（无限制）

```python
a = "python_java_c#_javascript"
list = a.split("_")
print(list)
#结果：['python', 'java', 'c#', 'javascript']

list = a.split("java")
print(list)
#结果：['python_', '_c#_', 'script']
```

3.4、**rsplit(sep=None, maxsplit=-1)**

指定分隔符对字符串进行分割并返回一个列表,只不过是**从字符串最后面逆序分割**,但列表仍是**由左到右开始索引**,第二个参数是分割几次

```python
url = "http://kr.shanghai-jiuxin.com/file/2022/0121/56af3130c05c83cf573fe5163115ace2.jpg"
name = url.rsplit("/",1)[1]
#56af3130c05c83cf573fe5163115ace2.jpg
```

**3.5 join()**合并，通常和列表连用

string.join(sequence)

Python中字符串是不可变对象，修改字符串就得将原字符串中的值复制，开辟一块新的内存，加上修改的内容后写入到新内存中，以达到“修改”字符串的效果。在使用“+”拼接字符串时，正是使用了重复性的复制、申请新内存、写入值到新内存的工作一遍遍的将字符串的值修改。而使用join()方法拼接字符串时，会先计算总共需要申请多少内存，然后一次性申请所需内存并将字符串复制过去。这样便省去了重复性的内存申请和写入，节省了时间消耗。

**4、查找和判断**

4.1、f查找

find(待查找内容) #有，返回子串位置；没有，返回-1

 		 index(待查找内容) #没有，则报错

print(待查找内容 in a) #是否在a中有。有返回True

print(待查找内容 not in a) #是否在a中有

4.2、判断

a.startswith("我家")#判断a对否以"我家"开头

a.endswith("高坡") #

a.isdigit() #判断是否是整数

**5 串驻留机制、字符串比较**is/id()

字符串驻留：仅保存一份相同且不可变字符 串的方法，不同的值被存放在**字符串驻留池**中。 注意：**字符串是在编译时进行驻留**，也就是说，如果字符串的值不能在编译时进行计算，将不会驻留。比如下面的例子：

```python
letter = 'd'
#以下全为运行时进行操作。
c = 'Hello Worl' + 'd'
d = 'Hello Worl' + letter
e = " ".join(['Hello','World'])

print(id(c))
print(id(d))
print(id(e))
'''
2940477095280
2940483917552
2940483917488
'''
```

Python支持字符串驻留机制，对于符合标识符规则的字符串(**仅包含下划线( _ )、字母**
**和数字**)会启用字符串驻留机制驻留机制。

1. 整数：[-5, 256] 之间的**整数**，会 预加载（缓存）到内存 ；
2. 符合命名规范的字符串；
3. 长度为0或1的字符串；

在交互环境上

```python

# 30 在 [-5， 256] 之间会驻留
a = 30
b = 30
print(a is b) # True

# -6 不在 [-5， 256] 之间不会驻留
a = -6
b = -6
print(a is b) # False

# wang 符合命名规则，会驻留
a = 'wang'
b = 'wang'
print(a is b) # True

# @qq.com 有特殊字符，不会驻留
a = '@qq.com'
b = '@qq.com'
print(a is b) # False

# 字符串长度为1，会驻留，即便不符合命名规则
a = '!'
b = '!'
print(a is b) # True
```

非交互环境，集成开发环境Pycharm：

```python
全为Ture
```

**6 成员字符串**

in / not in

**7 format()格式化**

Python2.6开始，新增了一种格式化字符串的函数str.format(),它增强了字符串格式化
的功能。
基本语法是通过{}和:来代替以前的%。.
format()函数可以接受不限个参数，位置可以不按顺序。

```python
a = "sasd{0}asd{1}"
b = a.format("zhang",2)
print(b)

d = "sasd{name}asd{id}".format(name = "zhang", id = "111")
print(d)

name = "san"
id = "000"
e = f"sasd{name}asd{id}"
print(e)

'''
sasdzhangasd2
sasdzhangasd111
sasdsanasd000
'''
```

**填充和对齐**

字符串填充常跟对齐一起使用。

1） ^、<、>分别是居中、左对齐、右对齐

2）对齐后面带宽度

3）:号后面带填充的字符，只能是一个字符，不指定的话默认是用空格填充。

```python
print("i am a {0:*^15}.".format("doctor"))
'''
i am a ****doctor*****..
'''
```

数字格式化：

![1690097512177]({{ '/assets/learning/01_python/1690097512177.png' | relative_url }})

**8 可变字符串**

在Python中，字符串属于不可变对象，不支持原地修改,如果需要修改其中的值,智
能创建新的字符串对象。但是,经常我们确实需要原地修改字符串,可以使用io.StringIO
对象或array模块。

```python
s = "char"
import io
s_io = io.StringIO(s)
print(s_io)
# seek()移动s指针
s_io.seek(1)
s_io.write(" hh hdfsh")
print(s_io.getvalue())
print(s_io)
'''
<_io.StringIO object at 0x000001D81AF59168>
c hh hdfsh
<_io.StringIO object at 0x000001D81AF59168>
'''
```

## 2.4 列表（中括号[ ]）

				list = ["张三","李四",王五","陈刘","张七"]

**4.1、定义和特性**

定义：是能装东西的东西，[ ]来表示一个列表，列表元素由','隔开，数据类型可不同

特性：

1、索引和切片。（索引超过范围报错）

2、for循环实现**遍历**，其本身可迭代

```python
	for item in list:
		print(item)
```
3、len()

`print(len(list))`

**4.2、增删改查**

**1、增加**

list.**append**("……")：追加 ,会将append的数据直接插入进去。

```python
a = [1,2,3]
a.append([1,2,3])
'''
[1, 2, 3, [1, 2, 3]]
'''
```

list.**insert**(index，"……")按位置插;

list.**extend**(list1)合并，会将extends的数据拆分放入。

```python
a = [1,2,3]
a.extend([1,2,3])
a
'''[1, 2, 3, 1, 2, 3]'''
```

**2、删除**

ret = list.**pop**(index) #给出被删除的索引，返回待删除元素;

list.**remove**("……") #删除某个元素

**3、修改**

list[1] = "……" #对该索引元素实现修改

**4、查询**

print(list[2])直接**索引查询**

```python
list = ["张1","张2","黄1","张3"]
for i in range(len(list)):
 item = list[i]
 if item.startswith("张"):
 new_name = "肖"+item[1:]
 list[i] = new_name
 print(list[i])
print(list)
```

4.3、**补充**

1、排序

list.sort() #升序排序；lsit.sort(reverse = True)# 降序排序

2、嵌套(与字典dict的嵌套差不多)

```python
list = ["张三",["李四",["王五","陈刘"]],"张七"]
print(list[1][1][0]) #第1项中的第1项的第0项
#输出：王五
```

**字符串的修改**对**原字符串**时**不发生改变**的。可通过另一字符串变量接收对原字符串的操作

```python
list = ["张三",["李四",["王五","陈刘",'python']],"张七"]
list[1][1][2] = list[1][1][2].upper()
print(list)
```

**3、列表的循环删除**

列表删除，找到待删除元素，若直接list.remove(item)会删不干净。因为列表删除了第一个待删除后，**后面的元素会来填补当前元素被删除位置**，依次前移，以至于删不干净。

循环列表本身删自身有问题

```python
tmp = [] #临时列表，存储待删除内容
for item in list:
 if item.startswith("张"): 			 #list.remove(item) #删不干净
 tmp.append(item)
for item in tmp:
 list.remove(item) #去原列表删除，每次遍历查找到item，删除。
 print(list)
```

## 2.5 元组 tuple （小括号（））

1、定义：**tuple元组 :不可变的列表**

**不存在增删改；查询可以索引查询**

```python
t = ("张三","李四","王五","陈刘","张七")
print(t)
print(t[1:3])
```

2、**元组若只有一个元素**，()默认是优先级，**type为字符串**

```python
t = ("张三")
print(t)
print(type(t))
#输出：
张三
<class 'str'>
```

3、元组不可变的坑：其中的列表元素可改变内容

```python
t = ("张三","李四",["王五","陈刘","张七"])
t[2].append("王四")
print(t)
#输出：t[2]为列表，可以改变其内容，但其实际起始地址并未变
('张三', '李四', ['王五', '陈刘', '张七', '王四'])
```

## 2.6 集合set (大括号{ ……} )

6.1、定义：set集合无序，不可重复，存储顺序无序，没有索引，**set集合要求存储数据必须是可以哈希的，不可变数据**。s = {} 为dict类型

**可哈希数据，不可变数据：数字, str,tuple,bool,function等**

**不可哈希数据，可变数据：list,dict,set**

由于可哈希性要求对象在其生命周期内保持不变，因此大多数（实际上几乎全部）内置的可哈希类型都是不可变的。这是因为如果对象可变，其哈希值可能会在对象状态改变后变得不一致，从而破坏哈希表（如集合 `set` 和字典 `dict`）的可靠性。

```python
s1 = str() #创建空字符串
l = list() #创建空列表
t = tuple()#创建空元组
s2 = set() #创建空集合
```

set()可以实现去重，将字符串转为一个个字符。

```python
set('shehgsdfsafds')
# {'a', 'd', 'e', 'f', 'g', 'h', 's'}
```

6.2增删改查

1、增加add( )、update(*s:Iterable)

```python
#add的参数必须为可hash数据
s2.add("张三")
s2.add("李四")
print(s2)
#update参数可以任意，底层会拆分成一个个元素，不可hash-->可hash
s.update({200,400,300})
s.update([100,123,456])
s.update((12,52,48))
```

2、删除

**①remove()**

一次删除一个指定元素，若指定元素不存在抛出KeyError

**②discard()**

一次删除一个指定元素，若指定元素不存在不抛异常

**③pop()**

一次删除一个任意元素

**④clear()**

清空集合

```python
s2.remove("张三")
print(s2)
```

3、修改：先删除，在添加

```python
s2.remove("张三")
s2.add("王麻子")
print(s2)
```

4、查

for item in s2:

5、**判断 in/not in**

print(10 in s)

6、交集、并集、差集

```python
s1 = {"张三","王五","陈刘"}
s2 = {"张三","李四","王五","陈刘","张七"}
print(s1 & s2)#交集
print(s1 | s2)#并集
print(s1 - s2)#差集
```

原集合有的元素，若add相同元素，不会插入。因此集合有一个重要作用：**去除重复**

```python
lst = ["张三","李四","张三","李四","张七"]
print(lst)
print(set(lst))#列表转集合去重print(list(set(lst))) #去重后，转为列表，但已无序

#输出
['张三', '李四', '张三', '李四', '张七']
{'张三', '张七', '李四'}
```

尽管位置不同，但两者比较是相同的，列表则不行。

```python
a = {'1234','12','34'}
b = {'12','1234','34'}
if ab:
 print("True")
'''True'''
```

## 2.7 字典dict{}（Json）

**7.1、定义**：字典以键值对形式进行存储{key1:value1,key2:value2,……}, key相当于索引。

key必须是可哈希的数据类型，即不可变。value可以是任何数据类型

**7.2、增删改查**

1、添加

```python
dic = dict() #空字典
dic = dict("tom"="爱喝汤","tony"="爱喝汤" ) #非空字典
dic['jay'] = "张三"
dic[1] = "李四"
print(dic)
dic['jay'] = "王麻子" #此时对关键字jay的value修改
dic.setdefault("tom","爱喝汤") #设置默认值print(dic)
```

2、删

```python
dic.pop("jay")
print(dic)
```

3、查

```python
print(dic["jay"]) #如果key不存在，报错print(dic.get("jay"))#如果key不存在，返回None
a = None #空对象 <class 'NoneType'>
a = "" #空字符串
```

**7.3、循环**

```python
dic = {"张三":"四岁",
		"李四":"五岁",
		"王五":"六岁",
		}
```

1、for循环

```python
#for循环直接拿到 key
for key in dic:
	print(key,dic[key])
```

2、所有key保存到列表中 .keys()

```python
print(list(dic.keys()))
#输出：
['张三', '李四', '王五']
```

3、所有value都放在列表中 .values()

```python
print(list(dic.values()))
#输出：
['四岁', '五岁', '六岁']
```

4、直接拿到key:value .items()返回**元组列表**

```python
print(list(dic.items()))
#输出：列表中元素为元组
[('张三', '四岁'), ('李四', '五岁'), ('王五', '六岁')]
```

```python
for item in dic.items():
	key = item[0]
	value = item[1]
	print(key,value)
#输出：
张三 四岁
李四 五岁
王五 六岁
```

### 解构（解包）：元组或列表可执行该操作

“

```python
#通过解包简化上述代码
for item in dic.items():
	key,value = item
 print(key,value)
#再次以解包简化操作，直接拿到字典所有的key:value

```

```python
#重要重要
for key,value in dic.items():
    print(key,value)
#输出：
张三 四岁
李四 五岁
王五 六岁
```

**使用zip进行并行迭代**

我们可以通过zip( )函数对多个序列进行并行迭代. zip( )函数在最短序列“用完”时就会停止。

```python
a1 = ("a1","a2")
b1 = ("b1","b2")
c1 = ("c1","c2")
for a,b,c in zip(a1,b1,c1):
	print("{0}-{1}-{2}".format(a,b,c))
for i in range(2):
	print("{0}-{1}-{2}".format(a1[i],b1[i],c1[i]))

'''
a1-b1-c1
a2-b2-c2
a1-b1-c1
a2-b2-c2
'''
```

**7.4、嵌套**

```python
zhangsan = {
    "name": "张三",
    "age": 26,
    "wife": {
        "name": "李四",
        "age": 26,
        "assistant":{
            "name": "王五",
            "age": 26,
        }
    },
    "children": [
        {"name":"张1","age":6},
        {"name": "张2","age": 2},
    ],
}
#张三妻子的助手的名字
print(zhangsan["wife"]["assistant"]["name"])
#张三第二个孩子明年多大岁数
print(zhangsan["children"][1]["age"] + 1)

#输出
王五
3
```

7.5、补充 删除指定元素，循环字典本身删自身有问题

```python
for key in dic:
    if key.startswith("张"):
    temp.append(key)

for key in temp:
	dic.pop(key)	
print(dic)
#输出
{'李四': '五岁', '王五': '六岁'}
```

## 2.8 bytes

1、字符和编码

总结:
1. ascii: 8bit, 1byte
2. ANSI =>一套标准，每个字符16bit, 2byte => 65536
3. **gbk**: 国标码16bit ,**2byte** **windows默认**
4. unicode: 万国码32bit, 4byte(没法用， 只是一个标准)
5. utf-8: **mac默认**
英文:
8bit, 1byte
欧洲: 16bit, 2byte
**中文: 24bit, 3byte**。gbk和utf-8不能直接进行转化。
6. 对于英语字母，UTF-8 编码和 ASCII 码是相同的, 所以 UTF-8 能兼容 ASCII 编码，这也是互联网普遍采用 UTF-8 的原因之一

2、编码和解码

```python
# bytes类型：b'*'
```

2.1、编码

```python
s = "张三"
bs1 = s.encode("gbk")
bs2 = s.encode("utf-8")
print(bs1)
print(bs2)
#输出
b'\xd5\xc5\xc8\xfd'
b'\xe5\xbc\xa0\xe4\xb8\x89'
```

2.2、解码

```python
#怎么把gbk转为utf-8
bs = b'\xd5\xc5\xc8\xfd'
#1、先转为文字符号(字符串)
s = bs.decode("gbk") #解码
bs2 = s.encode("utf-8")#编码
print(bs2)
```

```python
s = "abc"
print(s.encode("utf-8"))
#输出。
b'abc'
英文符合ASCII码能直接显示
如a:
ASCII:01000001
utf-8:00000000 01000001
jbk :00000000 01000001
编码时第二字节全零，因此与01000001相同，仍能直接表示
```

# 3 函数

## 3.1 函数概念

对某一个特定功能后者代码块进行封装，在需要使用该功能的时候直接调用。

## 3.2 函数的定义和调用

```python
#定义
def 函数名(参数1,……，参数n):
	函数体(封装的功能或代码块)
#执行
函数名(参数1,……，参数n)
```

## 3.3 函数的参数

**1、实参：**

1、**位置参数**：按照位置传参

```python
#执行
函数名("…","…",……,"…")
```

2、**关键字参数**，按照关键字名字传递参数

```python
#执行
函数名(参数3 = "",参数1 = "")
```

3、混合参数，**位置参数放前，关键字参数放后面**。否则，报错。这样用得不多，如：

```python
with open("11.jpeg",mode="rb") as f1:
```

**2、形参**

1、位置参数：按照位置一个个去声明变量

2、**默认值参数**：给形参一个默认值，如果调用函数时，实参不传递值，该默认值才生效，否则不生效。 **在函数定义时，第一个带有默认值的参数之后的所有参数都必须有默认值，否则，运行时报错。**

```python
def means(food,cai,tang,tian="iceberg"): #函数体
```

**形参顺序**：位置参数在前，默认值参数在后

- Python 的默认参数对象**在函数定义时就被创建并绑定**，而不是每次调用时创建
- 这意味着所有使用默认参数的调用都会共享同一个对象

```python
def append_to(element, lst=[]):
    print(id(lst))
    lst.append(element)
    return lst

print(append_to(1)) # 输出: [1]
print(append_to(2)) # 输出: [1, 2] (不是[2]!)

def append_to2(element, lst=None):
    print(id(lst))
    if lst is None:
    lst = [] # 每次都创建一个新列表
    lst.append(element)
    return lst

print(append_to2(1)) # 输出: [1]
print(append_to2(2)) # 输出: [2]

输出：
2980482216256
[1]
2980482216256
[1, 2]
140713508841712
[1]
140713508841712
[2]
```

底层原理

- **函数定义时**：

```
   def f(a=[]): pass
```

Python 会创建一个 `[]` 对象，并将其存储在函数的 `__defaults__` 属性中

- **函数调用时**：
如果不提供参数，Python 会使用 `__defaults__` 中的那个列表对象

对于任何可变默认参数，都应该使用这种模式：

```python
def func(arg=None):
    if arg is None:
    arg = mutable_default_value # 如 [], {}, set()
 # 函数主体
```

**3、动态传参**(传递参数个数不限)

1、*** args** 表示**位置参数的动态传参**，接收到的数据统一放进一个**元组**中

```python
def means(*food):
 print(food)
means("白米饭",'红烧如',"鸡蛋汤","哈根达斯")
#输出
('白米饭', '红烧如', '鸡蛋汤', '哈根达斯')
```

2、** kwargs 表示接收**关键字参数**的动态传参，接收到的所有关键字参数会被处理成**字典**

```python
def means(**food):
    print(food)
means(fu ="白米饭",tang = "鸡蛋汤")
#输出
{'fu': '白米饭', 'tang': '鸡蛋汤'}
```

**顺序：位置参数> * > 默认值 > ****

此时，默认值的修改只能通过关键字参数传参。上述参数可以随意搭配。

4、补充

* *在实参上，把列表stu_lst打散成位置参数传递
* **在实参上，把字典stu_lst打散成关键字参数传递

```python
stu_lst = ["韩立","南宫婉","厉飞雨"]
stu_lst1 = {"男主":"韩立","女主":"南宫婉"}
def func1(*args):
    print(args)
func1(*stu_lst) # * 在实参上，把列表stu_lst打散成位置参数传递
#输出 ('韩立', '南宫婉', '厉飞雨')
def func2(**args):
	print(args)	
func2(**stu_lst1)# **在实参上，把字典stu_lst打散成关键字参数传递
#输 {'男主': '韩立', '女主': '南宫婉'}
```

## 3.4 返回值

关于return

1、没有return，默认return None

2、有 return

只写return，程序停止，后续代码不再执行，返回None——>相当于break

**return 值。此时函数只返回一个值**

return 值1，值2,……。此时返回多个值，这些值存入**元组**中

## 3.5 函数对象--内存分析

**Python中“一切都是对象”**（对象放于堆区中，定义时一次创建多次使用，不会反复创建。指向堆中对象的变量放在栈区中），执行def定义函数时，实际系统就创建了相应的**函数对象**。

```python
# 函数对象
# test01为函数名，是存在栈中的一个变量，指向test01函数对象（在堆中），存放为函数对象的地址
def test01():
    print("test01")
test01()
#定义变量c,将test01的值赋值给c，即test01函数对象的地址赋值给c
c =test01
# c()表时调用变量所指向的函数对象。
c()
```

## 3.6 变量的作用域（全局、局部变量）

变量起作用的范围称为变量的作用域,不同作用域内同名变量间互不影响。变量分为:全
局变量、局部变量。

全局变量:

1.在函数和类定义之外声明的变量。作用域为定义的模块,从定义位置开始直到模块结束。

2.全局变量降低了函数的通用性和可读性。应尽量避免全局变量的使用。

3.全局变量-般做常量使用。

4.函数内要改变全局变量的值,使用global声明-下

局部变量:

1.在函数体中(包含形式参数)声明的变量，

2.**局部交量的引用比全局变量快,优先考虑使用**。

3.如果局部变量和全局变量同名,则在函数内隐藏全局变量,只使用同名的局部变量

**栈帧（stack free）**

每当函数被调用时，python会在栈区创建一个**栈帧**，用于**放置函数的局部变量**，指向堆区对象的引用。

函数内部优先使用函数内部的局部变量，若使用全局变量，要在函数内用global声明与全局变量同名的变量，表示该变量为全局变量。

```python
a=3 #全局变量
def test():
    global a
    print(a)
    a =1
    print(a)
test()
print(a)
```

locals()：返回局部变量字典

locals()：返回全局变量字典

```python
a=3 #全局
def test0():
    c = 100
def test():
    b = 5
    test0()
    print(locals())
    print(globals())
test()

'''
输出：
{'b': 5}
{……，'a': 3, 'test0': <function test0 at 0x000001FAF42B1048>, 'test': <function test at 0x000001FAF42B10D8>}
'''
```

## 3.7 参数的传递

函数的传参：实参->形参。python中一切皆对象，所有的复制操作都是**引用的赋值**，因此，参数传递是**引用传递**，不是**值传递**。操作分两种：

1) 对“可变对象”进行“写操作”, 直接作用于原对象本身

2) 对“不可变对象”进行“写操作”, 会产生一个新的“对象空间”, 并用新的值填充这块空间。( 起到其他语言的“值传递”效果,但不是“值传递”)

**可变对象**：字典、列表、集合、自定义的对象等

**不可变对象**： 数字、字符串、元组、function 等

### 3.7.1 传递可变对象

```python
# 2参数的传递
b = [10,20]
def add(m):
    print("b_id:", id(m))
    m.append(10)
add(b)
print("b_id:",id(b))
print(b)
#输出
'''
b_id: 1570220298760
b_id: 1570220298760
[10, 20, 10]
'''

```

### 3.7.2 传递不可变对象

传递参数是不可变对象(例如: 数字、字符串、元组、布尔值)， 实际传递的还是对象的**引用**，1）**引用不可变**，2）**引用所对应的数据不可改变**。在”赋值写操作”时，由于不可变对象无法修改，**系统会新创建一个对象。

**传递不可变对象是浅拷贝**。传递参数是不可变对象(例如: int. float. 字符串、元组、布尔值) , 实际传递的还是对象的引用。但在"写操作"时,会**创建一个新的对象拷贝**。 这个拷贝使用的是 “浅拷贝”,
不是“深拷贝”。

以下函数中的n=n+200实际操作时3步：**1）计算对象数据100+200=300；2）创建一个的新对象，接收数据300；3）并将对象引用返回给变量n。**

![1689926763979]({{ '/assets/learning/01_python/1689926763979.png' | relative_url }})

```python
# 05、传递不可变对象是浅拷贝
a = 10
print("a_id",id(a))
def test(m):
    print("m_id", id(m))
    m = 0
    print(m)
    print("m_id", id(m))
test(a)
print(a)
'''
a_id 140732606086496
m_id 140732606086496
0
m_id 140732553067552
'''
```

**注意**：传递不可变对象时，不可变对象里面包含的子对象若含有可变对象，则可改变该可变子对象的数据。

![1689947748349]({{ '/assets/learning/01_python/1689947748349.png' | relative_url }})

```python
a = (10,20,[1,2])
# a = 10
print("a_id",id(a))
def test(m):
 print("m_id", id(m))
 m[2][0] = 999
 print(m)
 print("m_id", id(m))
test(a)
print(a)
'''
a_id 140732606086496
m_id 140732606086496
20
m_id 140732606086496
'''
```

### 3.7.3 对象的浅拷贝和深拷贝（重要）

浅拷贝：**不拷贝子对象的内容，只是拷贝子对象的引用**。

深拷贝：会连子对象的内容也全部拷贝一份，对子对象的修改不会影响源对象。

导入import copy。使用内置函数: copy(浅拷贝)、deepcopy(深拷贝)。

```python
import copy
a = [1,2,[4,3]]
# 浅拷贝
b = copy.copy(a)
print("a:",a)
print("b:",b)

b.append(30)
b[2].append(2)
print("浅拷贝")
print("a:",a)
print("b:",b)

'''
output:
a: [1, 2, [4, 3]]
b: [1, 2, [4, 3]]
浅拷贝
a: [1, 2, [4, 3, 2]]
b: [1, 2, [4, 3, 2], 30]
'''
```



```python
import copy
# 深拷贝
def deepCopy():
    a = [10,20,[5,6]]
    # 深拷贝
    b = copy.deepcopy(a)
    print("a:",a)
    print("b:",b)

    b.append(30)
    b[2].append(7)
    print("深拷贝")
    print("a:",a)
    print("b:",b)

deepCopy()
'''
output:
a: [10, 20, [5, 6]]
a: [10, 20, [5, 6]]
深拷贝
a: [10, 20, [5, 6]]
b: [10, 20, [5, 6, 7], 30]
'''
```

## 3.8 lambda表达式和匿名函数

**lambda表达式可以用来声明匿名函数**。lambda 函数是一种简单的、在同一行中定义函数
的方法。**lambda 函数实际生成了一个函数对象**。
lambda表达式只允许包含一个表达式 ,不能包含复杂语句, 该表达式的计算结果就是函数的返回值。
lambda表达式的基本语法如下:

						lambda 引用 arg1,arg2,arg3,…. : <表达式>

arg1,arg2,arg3为函数参数，<表达式>看成函数体

```python
# lambda a,b,c,d:a*b*c*d 为匿名函数
f = lambda a,b,c,d:a*b*c*d
def f_muly(a,b,c,d):
    return a*b*c*d
print(f(1,2,3,4))

b_ = [lambda a: a * 2, lambda b: b * 3]
print(b_[0](6))

f_ = [f, f]
print(f_[0](1, 2, 3))
'''
output:
24
12
'''
```

## 3.9 eval函数

功能:将字符串str当成有效的表达式来求值并返回计算结果。

语法: eval(source[, globals=None, locals=None) -> value

参数:

source : 一个Python表达式 / 函数compile()返回的代码对象

globals :可选。必须是dictionary

locals :可选。任意映射对象

eval()使用原因：

1）在编译语言里要动态地产生代码，基本上是不可能的，但动态语言是可以，意味着软件已经部署到服务器上了，但只要作很少的更改，只好直接修改这部分的代码，就可立即实现变化，不用整个软件重新加载。

2）在machin learning里根据用户使用这个软件频率，以及方式，可动态地修改代码，适应用户的变化。

```python
# 执行字符串
s = "print('hello')"
eval(s)
# 就近取变量
a = 10
b = 12
c = eval("a+b")
print(c)

# 参数二指定，执行字符串取参数区域
dict1 = dict(a=100,b=200)
eval1 = eval("a+b", dict1)
print(eval1)
'''
output:
hello
22
300
'''
```

可用**eval**来处理输入的**字符串-->数字**的转换

```python
# eval接收 函数compile()返回的代码对象
eval1 = eval(input("请输入数字: "))
print(type(eval1))
'''
output:
请输入数字: 12
<class 'int'>
'''
```

## 3.10 嵌套函数（内部函数）

在函数内部定义的函数，仅供外函数使用

一般在什么情况下使用嵌套函数 ?

1.封装-数据隐藏，外部无法访问“嵌套函数”。

2.贯彻DRY(Don' t Repeat Yourself) 原则
嵌套函数.可以让我们在函数内部避免重复代码。

3.闭包

```python
# 10 内部函数
def printName (isChinese, name, familyName) :
    def inner_print(a, b):
        # print(" {0} {1}".format(a, b))
        print(f" {a} {b}")
        if isChinese:
            inner_print (familyName, name)
        else:
            inner_print (name, familyName)
            printName(True,"三","张")
            printName(False,"Tom","David")
            '''
张 三
Tom David
'''
```

**11 nonlocal关键字**

nonlocal
用来声明外层（内部函数使用外层函数的变量）的局部变量。
global
用来声明全局变量。

```python
a =1
print("a:",a)
def outer():
 b = 10
 print("b:",b)
 def inner():
 nonlocal b
 b = 20
 global a
 a = 11
 inner()
 print("b_new:",b)
outer()
print("a_new:",a)
'''
a: 1
b: 10
b_new: 20
a_new: 11
'''
```

## 3.11 LEGB规则

Python在查找
名称
时，是按照LEGB规则查找的:
Local--> Enclosed--> Global--> Built in。

Local
指的就是函数或者类的方法内部

Enclosed指的是嵌套函数 ( -个函数包裹另一个函数,包)

Global指的是模块中的全局变量

Built in指的是 Python为自己保留的特殊名称。

## 3.12 内置函数

fString: **f动态修改字符串**

```python
query = input("百度请输入：")
url = f'https://www.baidu.com/s?ie=UTF-8&wd={query}'
```

切片：slice(var1,var2,var3)

```python
s = slice(1,5,2)
print("你好，张三"[s])

#输出
好张
```

字符串：forma()、ord()、chr()

```python
a = 2
print(format(a,"08b"))
#b:二进制 bin(a)，o:八进制 oct(a)，x:十六进制 hex(x)
#"08b"是八位的二进制

a = "中" #python内存使用编码时unicode
print(ord(a)) #中字在Unicodede中的区位码是20013
print(chr(20013)) #用码位显示文字

for i in range(65536):
    print(chr(i)+" ",end="")
```

时间

```python
import time
#指定时间戳
tim =1645795520
# # 获得当前时间时间戳
now = int(time.time())
# # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
time1 = time.localtime(tim)
timeArray = time.localtime(now)
otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
print(otherStyleTime)
```

## 3.13 异常

异常类的跟类是BaseException。Exception是其子类。

### 1、try-except

多个except，先子类后父类异常，可以在最后增加BaseException

```python
try:
    a = "1"
    b = 0
    # res = a / b
    res2 = int(a)/b
except TypeError:
    print("类型错误")
except ZeroDivisionError:
    print("除数不能为0")
except BaseException as e:
    # as将捕获到的异常对象赋值给e
    print(e) #division by zero
```

### 2、try…except…else

如果try中没有异常，则执行else，如果抛出异常，则执行except。

### 3、try…except…else…finally

finally无论怎样都会执行，常用来释放try中临时申请的资源

| 常见异常类型 | 描述 |
| ----------------- | ---------------------- |
| zeroDivisionError | |
| IndexError | |
| KeyError | dic映射中没有这个键key |
| NameError | 未声明/初始化对象 |
| SystaxError | python语法错误 |
| ValueError | 传入无效参数 |

如下int a = 10，会出现SystaxError，因为变量没有类型，这是个错误语法。

```python
int a =10
```

### 4、traceback模块

打印异常信息

```python
import traceback

finally:
    with open("d://a.txt","a","utf-8") as f
        traceback.print_exc(file=f)

```

### 5、with上下文管理

finally块由于是否发生异常都会执行，通常我们放释放资源的代码。其实，我们可以通
过with上下文管理，更方便的实现释放资源的操作。with结构：

with contex_expr [as var]:

				语句块

with.上下文管理可以自动管理资源，在with代码块执行完毕后自动还原进入该代码之前的
现场或上下文。不论何种原因跳出with块，不论是否有异常，总能保证资源正常释放。极
大的简化了工作，在文件操作、网络通信相关的场合非常常用。

### 7、自定义异常类

程序开发中，有时候我们也需要自己定义异常类。自定义异常类一般都是**运行时异常**， 通常继承Exception或其子类即可。命名一般以Error、Exception 为后綴。
在程序中手动设置异常时，使用 raise 语句即可，自定义异常由**raise**语句主动抛出。

raise 语句的基本语法格式为：

```python
raise [exceptionName [(reason)]]
```

raise 语句有如下三种常用的用法：

1. raise：单独一个 raise。该语句引发当前上下文中捕获的异常（比如在 except 块中），或默认引发 RuntimeError 异常。
2. raise 异常类名称：raise 后带一个异常类名称，表示引发执行类型的异常。
3. raise 异常类名称(描述信息)：在引发指定类型的异常的同时，附带异常的描述信息。

```python
>>> raise
''''''

>>> raise ZeroDivisionError
'''Traceback (most recent call last):
 File "<pyshell#0>", line 1, in <module>
 raise ZeroDivisionError
ZeroDivisionError'''

>>> raise ZeroDivisionError("除数不能为零")
'''Traceback (most recent call last):
 File "<pyshell#2>", line 1, in <module>
 raise ZeroDivisionError("除数不能为零")
ZeroDivisionError: 除数不能为零
'''

try:
    a = input("输入一个数：")
    #判断用户输入的是否为数字
    if(not a.isdigit()):
    raise ValueError("a 必须是数字")
except ValueError as e:
	print("引发异常：",repr(e))
'''输入一个数：a
引发异常： ValueError('a 必须是数字',)'''
```

以上，当用户输入的不是数字时，程序会进入 if 判断语句，并执行 raise 引发 ValueError 异常。但由于其位于 try 块中，因为 raise 抛出的异常会被 try 捕获，并由 except 块进行处理。

```python
class AgeError(Exception):
    def __init__(self, errorInfo):
        Exception.__init__(self)
        self.errorInfo = errorInfo

    def __str__(self):
    	return f"年龄输入错误{self.errorInfo}"

# print(locals())# {'__name__': '__main__',…………}

if __name__ '__main__':# Ture则模块是独立运行文件，可以独立测试代码
    age = eval(input("年龄："))
    if age<0:
    	raise AgeError(age)
    else:
    	print("age:",age)
'''
年龄：-1
Traceback (most recent call last):
 File "D:/01Postgraduate/study_code/python_study/function/my_exception.py", line 20, in <module>
 raise AgeError(age)
__main__.AgeError: 年龄输入错误-1
'''
```

## 3.14 Debug

| button | description |
| ------------- | ------------------------------------------------------------ |
| step over | 若当前执行的是一个函数，则会把这个函数当做整体一步执行完，不会进入这个函数内部. |
| step into | 若当前执行的是一个函数，则进入函数内部详细执行 |
| step out | 当单步执行到子函数内时，会执行完函数余下部分，跳出函数。 |
| run to cursor | 一直执行，跳到下一个断点处。光标也可以是一个断点 |

variables的“+”可以增加观察器，写入变量名，用于观察该变量的变化。

![1690271780942]({{ '/assets/learning/01_python/1690271780942.png' | relative_url }})

浏览帧：

调试器列出断点处，当前线程正在运行的方法，每个方法对应一个“栈帧”。最上面的是当前断点所处的方法。

# 4 类与对象

面向对象编程将数据和操作数据相关的方法封装到对象中， 组织代码和数据的方式
更加接近人的思维，从而大大提高了编程的效率。

Python完全采用了面向对象的思想，是真正面向对象的编程语言,完全支持面向
对象的基本功能，例如:继承、多态、封装等。
Python中, - -切皆对象。我们在前面学习的数据类型、函数等, 都是对象。
注: Python支持面向过程、面向对象、函数式编程等多种编程范式。

●面向对象思考方式

遇到复杂问题,先从问题中找名词(面向过程更多的是找动词) 。然后确立这些名词哪
些可以作为类。再根据问题需求确定的类的属性和方法,确定类之间的关系。

面向对象和面向过程的总结

1）都是解决问题的思维方式 ,都是代码组织的方式；2）解决简单问题可以使用面向过程；3）解决复杂问题，宏观上使用面向对象把握,微观处理上仍然是面向过程。

●对象的进化：

随着编程面临的问题越来越复杂,编程语言本身也在进化。从主要处理简单数据开始,，随着数据变多进化“数组”；数据类型变复杂，进化出了“结构体”；处理数据的方式和逻辑变复杂。进化出了“对象”，将不同的数据类型、方法放在一起。

基本数据->数组->结构体->对象。

Python中，
-切皆对象”。
, 类也称为"类对象” ，类的实例也称为“实例对象”

## 4.1 类的定义

class 类名：

类体

其中类名一般要首字母大写，多个单词采用驼峰命名。

```python
class Student:
    # 类属性
    native_attr = '大傻瓜'

    # 构造方法。self当前对象本身
    def __init__(self, name, age):
        # self.na 为实体属性，将局部变量name赋值给实体属性
        self.na = name #实体属性na
        self.age = age #实体属性age

    # 实例方法。self必须为第一个参数
    def have(self):
    	print("学生吃三餐")

    # 静态方法
    @staticmethod
    def eat():
    	print("@staticmethod注解修饰为静态方法")

    # 类方法
    @classmethod
    def drink(cls):
    	print("@classmethod注解修饰为静态方法")

s1 = Student("zhang",12)

# 类以外定义的为函数，类内部为方法
def take():
 print("大朗该吃药了")

```

## 4.2 对象的创建

对象的创建又称类的实例化。

语法：实例名 = 类名()

```python
stu1 = Student("zhang",12)
print(id(stu1))
print(type(stu1))
print(stu1)

print(id(Student))#类名是指向类对象的，类本身也是一个对象

#输出
# 2432224492872
# <class '__main__.Student'>
# <__main__.Student object at 0x000002364BE07D48>
# 2432216636264
```

## 4.2 构造方法_ __init__ _()

我们前面说过一个 Python对象包含三个部分: id( identity识别码)、type (对象类型)、
value (对象的值)。
现在，我们可以更进一步的说 。一个Python对象包含如下部分:

1. id ( identity识别码)

2. type (对象类型)

3. value (对象的值)：

(1)属性( attribute )

(2)方法( method )

创建对象,我们需要定义构造函数_ init__ ()方法。 构造方法用于执行”实例对象的初始化工
作”, 即对象创建后,初始化当前对象的相关属性，无返回值。

__ init_ __()的要点如下:_

1.名称固定，必须为:__ __init__ _()。

2.第一个参数固定,必须为: self。self指的就是刚刚创建好的实例对象。

3.构造函数通常用来初始化实例对象的实例属性

5.**__ init__ ()**方法 :初始化创建好的对象。初始化指的是:“给实例属性赋值”。

6.**__ new__()**方法: 用于创建对象，但我们一般无需重定义该方法。

创建对象s = Student("11",21)：会先调用object.__ new_ _(cls)创建对象，将建好的对象地址返回给 s；再**__ init__ ()**初始化创建好的对象。

注:Python中的self相当于C++中的self指针，JAVA和C#中的this关键字。Python中,
self必须为构造函数的第一一个参数 ,名字可以任意修改。但一般遵守惯例,都叫做self.

## 4.3 实例属性

**实例属性**是**从属于实例对象**的属性，也称为“实例变量”，实例对象存在实例属性则存在。他的使用有如下几个要点：

1、实例属性一般在 _ init__ ( )方法中通过如下代码定义:_

				self.实例属性名=初始值

2、在**本类的其他实例方法中**，也是通过self进行访问：

 				**self.实例属性名**

3、创建实例对象后，通过实例对象访问：

				obj01 =类名( )	#创建对象，调用_ init__ ( )初始化属性
	
				obj01.实例属性名=值
可以给已有属性赋值.也可以新加属性

![1690122174970]({{ '/assets/learning/01_python/1690122174970.png' | relative_url }})

## 4.4 实例方法

实例方法是从属于实例对象的方法。实例方法的定义格式如下:

				def 方法名(self [形参列表]) :
	
						函数体

方法的调用格式如下:

				对象.方法名([实参列表])

要点:
1.定义实例方法时，第一个参数必须为self。和前面一样，self指当前的实例对象。
2.调用实例方法时，不需要也不能给self传参。self 由解释器自动传参，**解释器会以类去调用实例方法**，并传递指向self（当前实例对象）的变量（存储的是self的引用）。

函数和方法的区别
；

1.都是用来完成一个功能的语句块, 本质-样。

2.方法调用时，通过对象来调用。方法从属于特定实例对象，普通函数没有这个特点。

3.直观上看，方法定义时需要传递self 。函数不需要。

![1690123679159]({{ '/assets/learning/01_python/1690123679159.png' | relative_url }})

对于实例方法`say_score()`，然后通过类`Student`来调用该实例方法，但必须显示地传入一个实例对象`a`作为参数。

其他操作:
1. **dir(obj**)可以获得对象的**所有属性、方法**
2. obj._ dict_ 对象的属性字典
3. pass空语句
4. isinstance (对象,类型)判断 “对象”是不是"指定类型”

## 4.5 类对象

我们在前面讲的类定义格式中，“class 类名:”。实际上,当解释器执行class语句时,
就会创建一个类名的类对象。一个类只有一个类对象。

我们可以看到实际上生成了一个变量名就是类名"Student" 的对象。我们通过赋值给新变
量Stu2 ，也能实现相关的调用。说明，确实创建了“类对象”。

```python
class Student:
	pass
Stu2 = Student
s3 = Stu2()
```

## 4.6 类属性

类属性是从属于"类对象”的属性，也称为"类变量”。由于类属性从属于类对象，可以**被所有实例对象共享**。

类属性的定义方式:

class 类名:

				类变量名=初始值

在**类中或者类的外面**。我们可以通过:“类名.类变量名” 来读写。

```python
class Student:
     # 类属性
     native_attr = '大傻瓜'

     # 构造方法。self当前对象本身
     def __init__(self, name, age):
         # self.name 为实体属性，将局部变量name赋值给实体属性
         self.na = name #实体属性na
         self.age = age #实体属性age
         # 使用类属性
         Student.native_attr = "primary"
```

![1690195666695]({{ '/assets/learning/01_python/1690195666695.png' | relative_url }})

**实例方法的代码仍在类对象中**。

## 4.7 类方法

类方法是从属于"类对象”的方法。类方法通过装饰器@classmethod来定义，格式如下:

@classmethod

def类方法名(cIs [, 形参列表]) :

				函数体

要点如下:

1.@classmethod必须位于方法上面一行

2.第一个cls必须有; cls指的就是"类对象”本身;

3.调用类方法格式: "**类名 .类方法名(参数列表)**”。**参数列表中，不需要也不能给cls传值。**

4.类方法中访问实例属性和实例方法会导致错误

5.子类继承父类方法时，传入cls是子类对象，而非父类对象

**实例方法访问实例属性，类方法访问类属性**。**实例对象可访问类方法，且不需传参cls，底层会将对象的类自动传入；但是类对象不可直接无参访问实例方法，因为一个类有很多对象，所以必须传参访问。**

```python
 # 类方法
 @classmethod
 def drink(cls):
 # 实例方法访问实例属性
 # 类方法访问类属性
 print(cls.native_attr)
```

## 4.8 静态方法

Python中允许定义与"类对象”无关的方法，称为“静态方法”。“静态方法”和在模块中定义普通函数没有区别，只不过"静态方法”放到了"类的名字空
间里面”,，需要通过“类调用”。静态方法通过装饰器@staticmethod来定义,格式如下:

@staticmethod

def 静态方法名([形参列表]) :

					函数体

要点如下:

1.@staticmethod必须位于方法上面一行

2.调用静态方法格式: "**类名.静态方法名(参数列表)**” .

3.**类方法和静态方法中访问实例属性（实例属性在对象中）和实例方法（实例方法第一个参数为self，有类时还不一定有对象）会导致错误**。

## 4.9 _del _方法（析构方法）和垃圾回收机制

__del_ 方法称为"析构方法”,，用于实现对象被销毁时所需的操作。比如:释放对象
占用的资源,例如:打开的文件资源、网络连接等。

Python实现自动的垃圾回收，当对象没有被引用时(引用计数器计数为0) ，由垃圾回收器调用_ del_ 方法。

也可以通过del语句删除对象。从而保证调用_ _del_ _ 方法。
系统会自动提供_ del_ 方法,一般不需要 自定义析构方法。

```python
class Test:
	def __del__(self):
		print("销毁{0}".format(self))

t0 = Test()
t1 = t0
t3 = Test()
del t1 #这里t1所指对象的引用计数为2，所以del1,计数器-1=1，仍不会执行__del__
print("程序结束")
# 程序结束，会删除对应的所有对象
'''
程序结束
销毁<__main__.Test object at 0x000001DF9EE1B088>
销毁<__main__.Test object at 0x000001DF9EE1BD88>
'''

```

## 4.1 0 _call _方法、可调用对象

定义了 __call _方法的对象，称为”可调用对象”,
即该对象可以像函数-样被调用。则obj()调用的就是对象中的_ _call _

```python
class Test:
    def __del__(self):
    	print("销毁{0}".format(self))

    def __call__(self, salary):
    	return salary

t0 = Test()
sa = t0(1000)#调用的就是对象中的__call__方法
```

## 4.1 1 方法没有重载

在其他语言中，可以定义多个重名的方法，只要保证方法签名唯一即可。 方法签名包含3 ;
个部分:方法名、参数数量、参数类型。

Python中方法的参数没有类型(调用时确定参数的类型) ，参数的数量也可以由可变
参数控制。因此, Python中是没有方法的重载的。定义一个方法即可有多种调用方式，相当于实现了其他语言中的方法的重载。

如果我们在类体中定义了多个重名的方法,只有最后一个方法有效。

建议:**不要使用重名的方法! Python中方法没有重载。**

## 4.1 2 方法的动态性

Python是动态语言，我们可以动态的为类添加新的方法,，或者动态的修改类的已有的方法。

```python
class Person:
    def work(self):
    	print("")

    def study(s):
    	print(f"{s}正在学习")
    def work(s):
    	print(f"{s}正在工作")
Person.f_study = study
Person.work = work
p = Person()
p.f_study() #这里解释器会解释为 Person.f_study(p)
p.work()
# p.f_study() #这里解释器会解释为 Person.f_study(p)
'''
<__main__.Person object at 0x000002668671BD88>正在学习
'''
```

## 4.1 3 "__"私有属性和私有方法（实现封装）

Python对于类的成员没有严格的访问控制限制，这与其他面向对象语言有区别。关于私有属性和私有方法，有如下要点:

1.通常我们约定，**两个下划线开头**的属性是私有的(private)。其他为公共的(public)。在属性名前加双下划线`__`（如`__attr`），这会触发**名称改写（Name Mangling）**，将属性名改为`_ClassName__attr`形式

2.类内部可以访问私有属性(方法)

3.类外部不能直接访问私有属性(方法)

4.类外部可以通过"__ 类名__ 私有属性(方法)名" 访问私有属性(方法)。

[注]方法本质上也是属性!只不过是可以通过( )执行而已。所以,此处讲的私有属性和公有属性,也同时适用私有方法和公有方法的用法。

```python
class Employee:
    def __init__(self,name,age):
        self.name = name
        self.__age = age

e = Employee("张三",12)
print(e.name)
try:
	print(e.__age)
except AttributeError as err:
	print(err)

# print(dir(e))
print(e._Employee__age)
'''
张三
'Employee' object has no attribute '__age'
12
'''
```

```python
#私有方法
class Employee:

 def __init__(self,name,age):
 self.name = name
 self.__age = age

 def __work(self):
 print("{0}私有方法".format(self.name))

e = Employee("张三",12)
try:
 e.__work()
except AttributeError as err:
 print(err)

# print(dir(e))
print(e._Employee__work())
'''
'Employee' object has no attribute '__work'
张三私有方法
None
'''
```

## 4.1 4 @property装饰器

@property可以将一个**方法的调用**方式变成“**属性调用**”
。一般用来给私有属性做getter、setter方法

```python
class Employee:
    def __init__(self,name,age):
        self.__name = name
        self.__age = age

        # def get_name(self):
        # return self.__name
        # def set_name(self, age):
        # if age<0:
        # print("输入有误")
        # else:
        # self.__age = age

    @property
    def age(self):
        return self.__age

        # @age.getter
        # def age(self):
        # return self.__age

    @age.setter #这里是给@property修饰的age(self)的age的setter
    def age(self, age):
        if age<0:
        print("输入有误")
        else:
        self.__age = age

e1 = Employee("zhang san",22)
print(e1.age)

e1.age = -1
print(e1.age)

e1.age = 11
print(e1.age)

'''
22
输入有误
22
11
'''
```

# 5 面向对象三大特征

Python是面向对象的语言,也支持面向对象编程的三大特性:继承、封装(隐藏)、多态。

●封装(隐藏)
。隐藏对象的属性和实现细节,只对外提供必要的方法。相当于将“细节封装起来”, 只
对外暴露“相关调用方法。通过前面学习的“私有属性、私有方法”的方式，实现“封装”. Python 追求简洁的
语法，没有严格的语法级别的"访问控制符”，更多的是依靠程序员自觉实现。

●继承。继承可以让子类具有父类的特性，提高了**代码的重用性**。
从设计上是一种增量进化,原有父类设计不变的情况下,可以增加新的功能,或者改进
已有的算法。

●多态。多态是指同一个方法调用由于对象不同会产生不同的行为。

## 5.1 继承

Python支持多重继承，一个子类可以继承多个父类。继承的语法格式如下:

class子类类名(父类 1[,父类2 , .. ]):

类体

如果在类定义中没有指定父类，则**默认父类是object类**。也就是说，object是所有类的父
类，里面定义了一些所有类共有的默认实现，比如:_ new_ _().

**定义子类时，须在其构造函数中调用父类的构造函数**。调用格式如下:

父类名.__ init__ (self, 参数列表)

构造函数中包含调用父类构造函数。根据需要,不是必须。子类并不会自动调用父类的 __ init_ (), 我们必须显式的调用它。

```python
class Person:
    def __init__(self, name, age):
    self.name = name
    self.age = age

    def age(self):
    print("nono")

class Student(Person):
    def __init__(self, name, age,sex):
    # 必须显式的调用构造方法
    Person.__init__(self, name, age) #需要显示传入self
    super().__init__(name, age) #不需要显示传入self
    self.sex = sex

s = Student("ss",12,"man")
# print(dir(s))
print(s.age)

#
```

B继承了A，因为没重写__ init__ ，会去隐式执行__ init__ (self,name)方法，这里的self传的是B对象。因此定义B对象需要传参。如果重写了__init__(self,name)，则不会去显示调用父类的__ init__ 。

```python
class A:
    def __init__(self,name):
        self.name = name
        print(name)

class B(A):
    # def __init__(self):
    # pass
    def say(self):
        # A.say(self)
        super().say()
        print("B:",self)

                # 这里B继承了A，因为没重写，会去执行__init__(self,name)方法，因此需要传参
                # 如果重写了__init__(self,name)，则不会去显示调用父类的__init__
                b = B("Zhang")
                '''Zhang'''
```

## 5.2 类成员的继承和重写

1.成员继承：子类继承了父类除构造方法之外的所有成员。私有属性仍继承，只不过不能直接访问。

2.方法重写：子类可以重新定义父类中的方法，这样就会覆盖父类的方法，也称为“重写”。

## 5.3 object 根类

通过类的方法mro()或者类的属性__ mro__ 可以输出这个类的继承层次结构。

```python
class A:
    pass
class B(A):
    pass
class C(B):
    pass

print(C.mro())
'''
[<class '__main__.C'>, <class '__main__.B'>, <class '__main__.A'>, <class 'object'>]
'''
```

**object类是所有类的父类**，因此**所有的类都有object 类的属性和方法**。我们显然有必要深入研究一下 object类的结构。对于我们继续深入学习Python很有好处。

方法实际也是属性，只不过这个属性的类型是method而已。

**重写__ str__方法**

print(obj )会调用__ str__方法。

```python
class Person:
    def __init__(self, name, age):
    self.name = name
    self.age = age
    # 重写继承object跟类的__str__
    def __str__(self):
    return f"名字{self.name}"

p = Person("zhang san", 12)
print(p)
'''
名字zhang san
'''
```

## 5.4 多重继承

Python支持多重继承，一个子类可以有多个“直接父类"。这样，就具备了“多个父
类”的特点。但是由于这样会被“类的整体层次”搞的异常复杂,尽量避免使用。

MRO( )

Python支持多继承，如果父类中有相同名字的方法，在子类没有指定父类名时，解释器将按照MRO ( Method Resolution Order ) :方法解析顺序
“从左向右”按顺序搜索。
。我们可以通过 mro( )方法获得
"类的层次结构” ，方法解析顺序也是按照这个"类的层次结构”寻找的。

Super()获得父类的定义

在子类中, 如果想要获得父类的方法时，我们可以通过super( )来做。
**super( )代表父类的定义（实现对父类代码的复用），不是父类对象，传递的self仍是当前子类的实例对象**。

语法如下：

```python
super(type[, object-or-type])
```

- type -- 类。
- object-or-type -- 类，一般是 self

在Child(Parent)类中，**super(Child,self)** 首先找到 Child 的父类（就是类 Parent），然后把类 Child 的对象转换为类 Parent 的对象(**上转型对象**)。**对于super()._ _init_ _ 会将子类的self对象传递到父类的_ __init__ _ 函数中，则会让子类能够继承到父类的init函数中的内容，其中定义的实例属性也会作为子类的属性了。**

Python 3 可以使用直接使用 **super().xxx** 代替 **super(Class, self).xxx**

```python
class A:
    def say(self):
    print("A:",self)

class B(A):
    def say(self):
    # A.say(self)
    super().say()
    print("B:",self)

B().say()
'''
A: <__main__.B object at 0x000001BB591B3B08>
B: <__main__.B object at 0x000001BB591B3B08>
'''
```

## 5.5 多态

多态是指同一个方法调用由于对象不同会产生不同的行为。

1）多态是方法的多态，属性没有多态。

2）多态的存在由连个必要条件：继承、方法重写。

```python
class Human:
    def eat(self):
    pass
class Chinese(Human):
    def eat(self):
    print("中国人用筷子吃饭")
class English(Human):
    def eat (self):
    print("英国人用叉子吃饭")

def manEat (m) :
    if isinstance (m, Human) :
    m. eat()
    else:
    print("不能吃饭")
manEat (Chinese())
manEat (English())
'''
中国人用筷子吃饭
英国人用叉子吃饭
'''
```

## 5.6 特殊方法和运算符重载

**运算符也是对象**。Python的运算实际是通过调用对象的特殊方法实现的。

| 方法 | 说明 | 举例 |
| ---------- | ---------- | -------------- |
| _init _ | 构造方法 | p = Person( ) |
| _del _ | 析构方法 | 对象回收 |
| _str _ | 打印输出 | print(**) |
| _call _ | 可调用对象 | p() |
| _len _ | 长度 | len(a) |
| _getitem _ | 索引运算 | a[key] |
| _setitem _ | 索引赋值 | a[key] = value |

| 运算符 | 调用方法 | 描述 |
| ------ | -------- | ---- |
| + | _add _ | 加法 |
| - | _sub _ | 减法 |
| | | |

## 5.7 特殊属性

Python对象中包含了很多双下划线开始和结束的属性，这些是特殊属性，有特殊用法。常见的特殊属性:

| 特殊属性 | 含义 |
| ---------------------- | -------------------- |
| obj._ _dict _ _ | 对象的属性字典 |
| obj._ class_ | 对象所属的类 |
| class._ bases_ | 类的基类元组(多继承) |
| class._ base_ | 类的基类 |
| class._ mro_ | 类层次结构 |
| class._ subclasses_ () | 子类列表 |
| | |

## 5.8 对象的浅拷贝和深拷贝

●变量的赋值操作

只是形成两个变量,实际还是指向同一个对象。

●浅拷贝

Python拷贝一般都是浅拷贝。拷贝时，对象包含的子对象内容不拷贝。因此。源对象和拷贝对象会引用同一一个子对象。

●深拷贝

使用copy模块的deepcopy函數，递归拷贝对象中包含的子对象。源对象和拷贝对象
所有的子对象也不同。

## 5.9 组合

"is-a”关系，我们可以使用“继承”。从而实现子类拥有的父类的方法和属性。"is-a"
关系指的是类似这样的关系:狗是动物, dog is animal。狗类就应该继承动物类。

"has-a"关系，我们可以使用“组合”, 也能实现一个类拥有另一个类的方法和属性。
has-a"关系指的是这样的关系：手机拥有CPU。MobilePhone has a CPU。

当实例对象含有特定的对象属性，则为has。

```python
class MobilePhone:
    def __init__(self, cpu, screen) :
    self.cpu = cpu
    self. screen = screen

class CPU:
    def calculate(self):
    print(" cpu对象: ", self)

class Screen:
    def show(self):
    print("screen对象:", self)

m = MobilePhone(CPU(), Screen())

# has--a
m.cpu.calculate()
m.screen.show()

'''
 cpu对象: <__main__.CPU object at 0x0000017651F054C8>
screen对象: <__main__.Screen object at 0x0000017651F6CB08>
'''
```

## 5.10 抽象类(Abstract Class)

抽象类(Abstract Class)是面向对象编程中的一个重要概念，Python 通过 `abc` 模块提供了对抽象类的支持。

抽象类是不能被实例化的类，它存在的目的是为了让其他类继承它并实现它的抽象方法。抽象类通常包含一个或多个抽象方法。

抽象类(Abstract Class)是面向对象编程中的一个重要概念，Python 通过 `abc` 模块提供了对抽象类的支持。抽象类是不能被实例化的类，它存在的目的是为了让其他类继承它并实现它的抽象方法。抽象类通常包含一个或多个抽象方法。

### Python 实现抽象类

Python 中使用 `abc` 模块来创建抽象类：

```python
from abc import ABC, abstractmethod

class Shape(ABC): # 继承ABC表示这是一个抽象类
    @abstractmethod
    def area(self):
    pass

    @abstractmethod
    def perimeter(self):
    pass
```

示例：

```python
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def query(self, sql):
        pass

    @abstractmethod
    def close(self):
        pass

class MySQLDatabase(Database):
    def connect(self):
        print("Connecting to MySQL database...")

        def query(self, sql):
            print(f"Executing MySQL query: {sql}")

            def close(self):
                print("Closing MySQL connection")

                # 使用
                db = MySQLDatabase()
                db.connect()
                db.query("SELECT * FROM users")
                db.close()
```

## 5.11 设计模式

**1.工厂模式**

实现了创建者和调用者的分离,使用专门的工厂类将选择实现类、创建对象进行统一的管理和控制。

```python
class CarFactory:
    def create_car(self, brand):
        if brand "Benz":
            return Benz()
        elif brand "BYD":
            return BYD()

class BYD:
    pass
class Benz:
    pass

factory = CarFactory()
c1 = factory.create_car("BYD")
c2 = factory.create_car("Benz")
print(c1,c2)
```

**2. 单例模式**

单例模式( Singleton Pattern )的核心作用是确保一个类只有一个实例,并且提供一个访问该实例的全局访问点。

单例模式只生成一个实例对象，减少**位置参数**了对系统资源的开销。当一个对象的产生需要比较多的资源，如读取配置文件、产生其他依赖对象时，可以产生一个”单例对象”，然后永久驻留内存中，从而极大的降低开销。

```python
class Myingleton:
    # 类属性
    __obj = None
    #
    __init__flag = True
    # 重写__new__
    def __new__(cls, *args, **kwargs):
        if cls.__obj None:
            cls.__obj = object.__new__(cls)
            return cls.__obj

    def __init__(self, name):
        if Myingleton.__init__flag:
            print("init-----")
            self.name = name
            Myingleton.__init__flag = False

            a = Myingleton("single")
            b = Myingleton("b")
            print(a)
            print(b)

'''
init-----
<__main__.Myingleton object at 0x0000016C78025488>
<__main__.Myingleton object at 0x0000016C78025488>
'''
```

# 6 文件操作

## 6.1 文件读写

按文件中数据组织形式，我们把文件分为文本文件和二进制文件两大类。

1.**文本文件**

文本文件存储的是普通“**字符**”文本，默认为**unicode字符集**（Python默认使用；2字节表一个字符），可以使用记事本程序打
开。但是，像word软件编辑的文档不是文本文件。

2.**二进制文件**

二进制文件把数据内容用“**字节**”进行存储，无法用记事本打开。必须使用专用的软件解码。常见的有: MP4视频文件、MP3音频文件、JPG 图片、doc 文档等等.

1、创建文件对象open()

open(文件路径, mode="", encoding="")

windows中文默认编码：gbk;mac中文默认编码：utf-8

| 文件路径：  | 1、绝对路径 D:/tes/ss                             |
| ----------- | ------------------------------------------------- |
|             | 2、相对路径。相对于当前程序所在的文件夹           |
|             | ../上一层文件夹                                   |
| mode：      | 模式                                              |
|             | 1、r: read 读取                                   |
|             | 2、w：write 写入，会清空原文档，不存在则新建      |
|             | 3、a：append 追加                                 |
|             | 4、b：bytes **读取非文本文件**(图片等)(rb,wb组合) |
|             | 5、+：读、写模式（可以其他模式组合）              |
| encoding    | 编码方式                                          |
| **newline** | **newline='' 去除空行**                           |

文本文件的写入一般就是三个步骤:
1.创建文件对象
2.
写入数据 3.关闭文件对象

![1690272744386]({{ '/assets/learning/01_python/1690272744386.png' | relative_url }})

f.close( )关闭OS和文件等资源。

1.1、打开open与关闭close

```python
f = open("text.txt",mode="r",encoding="utf-8")
#open为打开文件与客户端的一根管道，每次open打开一根管道，f为一个文件管道，每次读了一行，f下次会从下一行开始读取
f.close()#关闭管道
```

1.2文件的读取

```python
content = f.readlines()
print(content)
#输出 每行末位有换行符 成为一个列表，每行为一个列表元素
['张三\n', '李四\n', '王麻子']
```

```python
f.read()
```

```python
line = f.readline()
print(line)
line = f.readline()
print(line)
line = f.readline()
print(line)
#输出
张三

李四

王麻子
line = f.readline().strip()
print(line)
line = f.readline().strip()
print(line)
line = f.readline().strip()
print(line)
#输出 strip()去掉换行符
张三
李四
王麻子
```

**最重要的一个文本读取方式**

```python
for line in f:
	 print(line.strip())
```

1.3写入f.write()

```python
#w模式下，每次都会清空文档
f = open("text.txt",mode="w",encoding="utf-8")
#f.write("小米粥")
#f.write("\n")#换行
#f.close()#关闭管道

for item in list:
    f.write(item)
    f.write("\n")
f.close()#关闭管道
```

**2、with open() as上下文的使用，with程序块结束，自动关闭管道**

```python
with open("text.txt",mode="r",encoding="utf-8") as f:
    #f = open()
    for line in f:
    print(line.strip())
```

2.1、读取图片，**mode="rb"**

```python
with open("11.jpeg",mode="rb") as f:
    for line in f:
    print(line)
```

2.2 文件的复制

```python
#python是解释性语言，读一行，执行一行，“\”表示下一行与上一行是同一行代码。
with open("11.jpeg",mode="rb") as f1,\
    open("../22.jpeg",mode="wb") as f2:
    for line in f1:
    f2.write(line)
```

2.3、文件修改操作

**文件修改过程 1、从原文件读取内容；2、在内存中实现修改；3、把修改后的内容写入副本新文件；4、删除原文件，副本新文件改名为原文件名**

```python
import os #操作系统模块用于与硬件打交道
import time
#完成副本
with open("text.txt",mode="r",encoding="utf-8") as f1, \
    open("text_副本.txt", mode="w", encoding="utf-8") as f2:
    for line in f1:
    line = line.strip() #去掉换行
    if line.startswith("张"):
    line = line.replace("张","韩")#修改
    f2.write(line)
    f2.write("\n")
#让程序休眠3秒
time.sleep(3)
#删除原文件。
os.remove("text.txt")
#让程序休眠3秒
time.sleep(3)
#副本文件改名为原文件名
os.rename("text_副本.txt","text.txt")
```

enumerate()函数

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

```python
a=["张","海\n","大\n"]
b = enumerate(a)
print(a)
print (list(b))
c =[ temp.rstrip()+"#"+str(index) for index, temp in enumerate (a)]
print(c)
'''
['张', '海\n', '大\n']
[(0, '张'), (1, '海\n'), (2, '大\n')]
['张#0', '海#1', '大#2']
'''
```

## 6.2 pickle序列化

Python中，一切皆对象，对象本质上就是一个“存储数据的内存块”。有时候，我们需要将“内存块的数据”保存到硬盘上，或者通过网络传输到其他的计算机上。这时候，就需要“对象的序列化和反序列化”。对象的序列化机制广泛的应用在分布式、并行系统上。

序列指的是:将对象转化成“串行化（一串数据）”数据形式，存储到硬盘或通过网络传输到其他地方。反序列化是指相反的过程，将读取到的“串行化数据”转化成对象。

我们可以使用pickle模块中的函数，实现序列化和反序列操作。

序列化我们使用:

pickle.dump(obj, file)
obj就是要被序列化的对象，file 指的是存储的文件

pickle.load(file)
从file读取数据，反序列化成对象。**反序列化后的到的是一数据都相同的一个新对象**

```python
import pickle

a = (1,2,4,"huang")
print(a)
print("a_id:",id(a))
with open("data.dat","ab") as f:
    pickle.dump(a,f)

with open("data.dat","rb") as f:
    load = pickle.load(f)
    print(load)
    print("load_id:",id(load))
'''
(1, 2, 4, 'huang')
a_id: 1751194917208
(1, 2, 4, 'huang')
load_id: 1751192054264
'''

```

## 6.3 CSV文件的操作

csv(Comma Separated Values)是逗号分隔符文本格式，常用于数据交换、Excel 文
件和数据库数据的导入和导出。与Excel文件不同，CSV 文件中:

●
值没有类型，所有值都是字符串

● 不能指定字体颜色等样式

●不能指定单元格的宽高，不能合并单元格

●没有多个工作表

●不能嵌入图像图表

Python标准库的模块csv提供了读取和写入csv格式文件的对象。

```python
import csv
# 读取
with open("data/test01.csv", "r",encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
        '''
['1', '2', '3', '4']
['张三', '熊孩子', '明明', '字字']
'''

# 写入,以列表的方式写入
with open("data/test01.csv", "w",encoding="utf-8",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["0","1","2","3"])
    writer.writerow(["tang","long","yuan","feng"])

    inn = [["tang","long","yuan","feng"],["tang","long","yang","feng"]]
    writer.writerows(inn)
```

## 6.4 OS和os.path模块

os模块可以帮助我们直接对操作系统进行操作。我们可以直接调用操作系统的可执行
文件、命令，直接操作文件、目录等等。在系统运维的核心基础。

```python
import os

# os.system("notepad.exe")
# os.system("ping github.com")

# 直接调用可执行程序
os.startfile("d:wechat.exe")
```

1 文件和目录操作

os模块下常用操作文件的方法

| 方法名 | 描述 |
| ---------------- | ------------------------------ |
| remove(path) | 删除指定的文件 |
| rename(src,dest) | 重命名文件或目录 |
| stat(path) | 返回文件的所有属性 |
| listdir(path) | 返回path目录下的文件和目录列表 |

os模块下常用操作目录的方法

| 方法名 | 描述 |
| ---------------------------- | --------------------------------- |
| mkdir(path) | 创建目录 |
| makedirs(path1/path2/pat...) | 创建多级目录 |
| rmdir(path) | 删除目录 |
| removedirs(path1/path... | 删除多级目录 |
| getcwd() | 返回当前工作目录:current work dir |
| chdir(path) | 把path设为当前工作目录 |
| walk() | 遍历目录树 |
| sep | 当前操作系统所使用的路径分隔符 |
| | |

os.path模块

os.path模块提供了目录相关(路径判断、路径切分、路径链接、文件夹遍历)的操作

os.path.join(path, file) #文件名补全路径

![1691292238122]({{ '/assets/learning/01_python/1691292238122.png' | relative_url }})

## 6.5 shutil(拷贝和压缩)和zipfile

```python
import shutil
shutil.copyfile("data/test01.csv","data/01.csv")
# 拷贝目录中即其文件
shutil.copytree("data/电影","电影")
# 忽略扩展名为html，txt不拷贝
shutil.copytree("data/电影","电影",ignore=shutil.ignore_patterns("*.html","*.txt"))

# 压缩 压缩到"data/sister
shutil.make_archive("data/sister","zip","data/test01.csv")

```

```python
# 创建接收待压缩对象
z1 = zipfile.ZipFile("data/a.zip","w")
z1.write("file01.py")
z1.close()

# 解压
z2 = zipfile.ZipFile("data/a.zip","r")
z2.extractall("data")
z2.close()
```

# 7 模块与相关函数

## 7.1 模块化程序设计理念

量变引起质变。物以类聚。

程序越来越复杂,语句多了,怎么管理?很自然的,我们会将实现同一个功能的语句封
装到函数中,统-管理和调用,于是函数诞生了。

程序更加复杂,函数和变量多了,怎么管理?同样的思路，“物以类聚”, 我们将同一
类型对象的“数据和行为”, 也就是“变量和函数”, 放到起统-管理和调用，于是“类
和对象”诞生了。

程序继续复杂,函数和类更加多了,怎么办?好,我们将实现类似功能的函数和类统统
放到一个模块中，于是“模块”诞生了。

程序还要复杂，模块多了,怎么办?于是,我们将实现类似功能的模块放到-起，于
是“包"就诞生了。

![1690289985873]({{ '/assets/learning/01_python/1690289985873.png' | relative_url }})

1.Python程序由模块组成。-一个模块对应python源文件，--般后缀名是：.py.

2.模块由语句组成。 运行Python程序时，按照模块中语句的顺序依次执行。

3.语句是Python程序的构造单元,用于创建对象、变量赋值、调用函数、控制语句等。

Python有标准库模块、**第三方模块**（第三方上传到Python官网）。模块化是为了后期的代码复用、搭建、增强维护性。

**模块化编程的流程**

1.设计API ,进行功能描述。

2.编码实现API中描述的功能。

3.在模块中编写测试代码，并消除全局代码。

4.使用私有函数实现不被外部客户端调用的模块函数。

可以同过___doc__ _ 获得模块的**文档字符串**（''' 多行注释'''）的内容。

每个模块都有一个名称，通过特殊变量_
name_ 可以获取**模块的名称**。在正常情况下，模块名字对应源文件名。仅有一 个例外 ,就是**当一个模块被作为程序入口时**(主程序、交互式提示符下) ,它的**_ name_ 的值为”_ main_ .”** 。我们可以根据这个特点，将模块源代码文件中的测试代码进行独立的处理。例如:

import math

math._ name_ #输出'math'

## 7.2 模块的导入

### 1、import导入

import语句的基本语法格式如下:

import
模块名#导入一个模块

import
模块1 ,模块2...
#导入多个模块

import
模块名
as模块别名
#导入模块并使用新名字

import加载的模块分为四个通用类别:

a.使用python编写的代码( .py文件) ;

b.已被编译为共享库或DLL的C或C++扩展;

c.包好一组模块的包

d.使用C编写并链接到python解释器的内置模块;

我们一般通过import语句实现模块的导入和使用, import本质上是使用了内置函数**_**
**import_ ()**。
当我们通过import导入一个模块时, python解释器进行执行，最终会生成一个**对象**,
这个对象就**代表了被加载的模块**。

```python
import math
print(id(math))
print(type(math))
print(math.pi)
'''
2804874888120
<class 'module'>
'''
```

**一个模块无论导入多少次，这个模块在整个解释器进程内有且仅有一个实例对象。**

```python
import math #会执行
import math #不会再次执行
```

### 2、from…import…导入

import导入的是模块。from..import 导入的是模块中的一个函数/一个类。

如果进行类比的话，import导入的是"文件”，我们要使用该"文件”下的内容，必
须前面加“文件名称”。fom..import 导入的是文件下的"内容”, 我们直接使用这
些"内容”即可,前面再也不需要加"文件名称”了。

### 3、_import _()、importlib动态导入

import语句本质上就是调用内置函数import_ () , 我们可以通过它实现动态导入。给
_
import_()动态传递不同的参数值 ,就能导入不同的模块。

注意: 一般不建议我们自行使用_ import__ ()导入 ,其行为在python2和python3中
有差异,会导致意外错误。如果需要动态导入可以使用importlib模块。

```python
s = 'math'
m = __import__(s)
print(m.pi)

import importlib
ma = importlib.import_module("math")
print(ma.pi)
```

## 7.3 package包

1、当一个项目中有很多个模块时,需要再进行组织。我们将功能类似的模块放到一起,
形成了“包”
本质上，“包”就是一个必须有**_ init__ .py** 的文件夹。包下面可以包含"模块(module)”, 也可以再包含“子包(subpackage)"。典型结构如下:

![1690292330533]({{ '/assets/learning/01_python/1690292330533.png' | relative_url }})

导入包与导入模块一样。

1. from package import item这种语法中，item可以是包、模块，也可以是函数、
类、变量。
2. import item, item2 这种语法中，item必须是包或模块，不能是其他。

**2、******导入包的本质其实是"导入了包的_ init_ .py" 文件**。也就是说，”import pack1" 意味着执行了包pack1下面的_ init_ .Py 文件。这样 ，可以在_ init_ .py 中**批量导入**我们需要的其他标准、第三方模块，而不再需要一个个导入。

_ init_ .py的三个核心作用:

1.作为包的标识,不能删除。

2.用来实现模糊导入

3.导入包实质是执行_ init_ .py文件,可以在_ init_ .py 文件中做这个包的初始化、以及
需要统一执行代码。

### 3、用*模糊导入包

import * 导入包中所有子模块。

一个索引由_ init_ .py 定义**_ all_** 变量， 该变量为一列表，在包下的
_ init_ .py中，可定义_ all_ = ['module_A"," module_A2"]
这意味着，from aa import *会从对应的包中导入以 上两个子模块;

from .. import module_ A
#..表示上级目录

from . import module_ _A2
#.表示同级目录

### 4、sys.path和模块搜索路径

当我们导入某个模块文件时，Python 解释器去哪里找这个文件呢?只有找到这个文
件才能读取、装载运行该模块文件。它一般按照如下路径寻找模块文件(按照顺序寻找,找
到即停不继续往下寻找) :

1.内置模块

2.当前目录

3.程序的主目录

4.pythonpath目录(如果已经设置了)

5.标准链接库目录

6.第三方库目录( site-packages目录)

7..pth文件的内容(如果存在的话)

8.sys.path.append0临时添加的目录

当任何一个python程序启动时就将上面这些搜索路径(除内置模块以外的路径)进行收集，放到sys模块的path属性中( sys.path )。

## 7.4 变量的定义和使用（变量是用来指向对象）

对象(或者说变量，变量是用来指向对象，是对象的引用，如C指针)由三部分组成：

**标识**: 标识对象存储对应的**物理内存地址**（定义变量使也标识了对象的类型），可用内置函数id(obj)获取

**类型**：用内置函数type(obj)获取

**值**：print打印

### 注1：类型属于对象，也就是对于对象来说的，而不是属于变量

```python
# 变量名 --引用--> 对象
a=3
a="spam"
a=1.23
```

在python 中，变量名没有类型，我们只是把a修改为对不同的对象的引用。因为变量没有类型，我们实际上并没有改变变量a的类型，只是让变量引用了不同类型的对象而已。

对象知道自己的类型。每个**对象**都包含一个**头部信息**，其中标记了这个对象的类型，因为对象记录了他们的类型，变量就没有必要再记录了

#### 注2：同意运算符：操作符`is` 和``的区别与关系：

is

判断两个变量是否指向同一个对象，比较两个引用是否代表了内存中同一个地址，是否指向同一块内存空间。通过**id(obj)**返回值判断。

比较**对象的本身内容**{类型(type)和值(value)}是否一致。默认会调用对象的_eq__方法。

**关系**：

当id相同时，is 为True，两者的type和value肯定相同，

但当type与value相同时， 为True，但两者的id不一定相同。

## 7.5 变量类型

- 带默认值的参数

在Python中的类或者函数中，若参数在声明时附带了它的默认值，则在实例化或调用时，可以选择性地为该参数赋值，例如：

```python
#默认值参数
def foo_v1(a: int, b: int = 1):
    print(a + b)
    #未给b传入实参时，采用默认值
    foo_v1(2)

    # 输出
    # >>> 3

```

- Typing.Optional类

可选类型，作用几乎和带默认值的参数等价，不同的是使用Optional会告诉你的IDE或者框架：这个参数除了给定的默认值外还可以是None，而且使用有些静态检查工具如numpy时，对 a: int =None这样类似的声明可能会提示报错，但使用a :Optional[int] = None不会。

`Optional[T]`表示：

- 这个值可以是类型`T`
- 或者可以是`None`

它等价于`Union[T, None]`，但使用`Optional`更加清晰和简洁。`Union[T, None]` 是 Python 类型注解（Type Hint）中的一种联合类型表示法，它明确表示一个值可以是类型 `T` **或者** `None`。

```python
#Optional
from typing import Optional

def foo_v2(a: int, b: Optional[int] = None):
    if b:
        print(a + b)
    else:
        print("parameter b is a NoneType!")

        #只传入a位置的实参
        foo_v2(2)

        # 输出
        >>> parameter b is a NoneType!

```

## 7.6 None

“Comparisons to singletons like None should always be done with ‘is’ or ‘is not’, never the equality operators.”
— From PEP8

None在Python里是个`单例对象`，一个变量如果是None，它一定和None指向同一个内存地址。None是python中的一个特殊的常量，表示一个空的对象。空值是Python中的一个特殊值，`数据为空并不代表是空对象`，例如`[]，''，()，{}`等都不是None。

None和任何对象比较返回值都是False 。

## 7.7 as的三种用法

### 6.1.with...as...

第一种是和with结合使用，主要用于文件的读写操作，省去了关闭文件的麻烦。

### 2.导入模块起别名

第二种是导入模块是对模块进行重命名，也就是给模块起一个别名。

### 3.except结合使用

第三种是和except组合使用，将捕获到的异常对象赋值给e。

```python
except BaseException as e:
 print(e) #division by zero
```

## 7.8 random

random.shuffle()**用于将一个列表中的元素打乱顺序**，值得注意的是使用这个方法**不会生成新的列表，只是将原列表的次序打乱**。



## 7.9 迭代器与生成器

### 7.9.1 迭代器

迭代器有两个基本的方法：

**iter( 可迭代对象)** :创建迭代器对象

**next()**: 输出迭代器的下一个元素

```python
list=[1,2,3,4]
it = iter(list) # 创建迭代器对象
for x in it:
    print(x, end=" ")
    '''1 2 3 4'''
```

```python
import sys # 引入 sys 模块
list=[1,2,3,4]
it = iter(list) # 创建迭代器对象
while True:
    try:
        print (next(it))
    except StopIteration:
        sys.exit()
```

### 7.9.2 创建一个迭代器

迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。

把一个类作为一个迭代器使用需要在类中实现两个方法 __iter__() 与 __next__() 。

__iter__() 方法返回一个特殊的迭代器对象， 这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成。

__next__() 方法（Python 2 里是 next()）会返回下一个迭代器对象。

```python
class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        if self.a <= 5:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration

            myclass = MyNumbers()
            myiter = iter(myclass)

            # 每次迭代中会去执行next()方法
            for x in myiter:
                print(x)
'''1
2
3
4
5'''
```

### 7.9.3 生成器函数 yield

在 Python 中，使用了 **yield** 的函数被称为生成器（generator）。

**yield** 是一个关键字，用于定义生成器函数，生成器函数是一种特殊的函数，可以在迭代过程中逐步产生值，而不是一次性返回所有结果。

跟普通函数不同的是，**生成器是一个返回迭代器的函数**，只能用于迭代操作，更简单点理解生成器就是一个迭代器。

当在生成器函数中使用 **yield** 语句时，函数的执行将会暂停，并将 **yield** 后面的表达式作为当前迭代的值返回。

然后，每次调用生成器的 **next()** 方法，**send()**方法或使用 **for** 循环进行迭代时，**函数会从上次暂停的地方继续执行，直到再次遇到 yield 语句，并返回对应的数据**。这样，生成器函数可以逐步产生值，而不需要一次性计算并返回所有结果。

**调用一个生成器函数，返回的是一个迭代器对象**。

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

        # 创建生成器对象
        generator = countdown(5)

        # 通过迭代生成器获取值
        print(next(generator)) # 输出: 5
        print(next(generator)) # 输出: 4
        print(next(generator)) # 输出: 3

        # 使用 for 循环迭代生成器
        for value in generator:
            print(value) # 输出: 2 1
```

通过创建生成器对象并使用 next() 函数或 for 循环迭代生成器，我们可以逐步获取生成器函数产生的值。

生成器函数的优势是它们可以按需生成值，避免一次性生成大量数据并占用大量内存。

```python
def foo():
    print("starting...")
    while True:
        res = yield 4
        print("res:",res)
        g = foo()
        print(next(g))
        print("*"*20)
        print(g.send(7))
        '''starting...
4
********************
res: 7
4'''
```

生成器.send()函数：启动生成器，发送数据至生成器中。

## 7.10 repr() 函数

将对象转化为供解释器读取的形式， 返回一个**对象的 string 格式**。 格式如下：

```python
repr(object)
```

```python
s = 'RUNOOB'
repr(s)
#'RUNOOB'
dict = {'runoob': 'runoob.com', 'google': 'google.com'};
repr(dict)
#{'google': 'google.com', 'runoob': 'runoob.com'}

s="物品\t单价\t数量\n包子\t1\t2"
print(s)
print(repr(s))
物品 单价 数量
包子 1 2
'物品\t单价\t数量\n包子\t1\t2'
```

## 7.11 assert

在没完善一个程序之前，我们不知道程序在哪里会出错，与其让它在运行时崩溃，不如在出现错误条件时就崩溃。

语法格式如下：

assert expression

等价于：

```python
if not expression:
     raise AssertionError
```

assert 后面也可以紧跟参数:

```python
assert expression [, arguments]
```

等价于：

```python
if not expression:
	raise AssertionError(arguments)
```

```python
def zero(s):
    a = int(s)
    assert a > 0,"a超出范围" #这句的意思：如果a确实大于0，程序正常往下运行
    return a

zero("-2") #但是如果a是小于0的，程序会抛出AssertionError错误，报错为参数内容“a超出范围”

```

结果：

```python
Traceback (most recent call last):
 File "e:\Python_list\class_student\temp.py", line 6, in <module>
 zero("-2")
 File "e:\Python_list\class_student\temp.py", line 3, in zero
 assert a > 0,"a超出范围"
AssertionError: a超出范围
```

## 7.12 all( )

all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False。

元素除了是 0、空、None、False 外都算 True。

```python
>>> all(['a', 'b', 'c', 'd']) # 列表list，元素都不为空或0
True
>>> all(['a', 'b', '', 'd']) # 列表list，存在一个为空的元素
False
```

## 7.13 map( )

**map( )** 会根据提供的函数对指定序列做映射。

第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。

map() 函数语法：

```python
map(function, iterable, ...)
```

- function -- 函数
- iterable -- 一个或多个序列，每个序列中的每个元素对应函数需要的参数，都会进入函数。

```python
>>> def square(x) : # 计算平方数
    ... return x ** 2
...
>>> map(square, [1,2,3,4,5]) # 计算列表各个元素的平方
[1, 4, 9, 16, 25]
>>> map(lambda x: x ** 2, [1, 2, 3, 4, 5]) # 使用 lambda 匿名函数
[1, 4, 9, 16, 25]

# 提供了两个列表，对相同位置的列表数据进行相加
>>> map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
[3, 7, 11, 15, 19]
```

## 7.14 ->

```python
def str2bool(v: str) -> bool:
    v = v.lower()
    if v "true":
        return True
    elif v "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")
```

'->'用于指定函数的**返回类型注解**。它表示函数str2bool的返回值的类型为bool。

在Python中，类型注解是一种用于标注函数参数和返回值类型的语法。它不会对代码的执行产生任何影响，仅仅是为了提供类型提示和文档说明的作用。。

## 7.15 argparse

argparse模块是命令行选项、参数和子命令解析器。可以让人轻松编写用户友好的命令行接口。适用于代码需要频繁地修改参数的情况。

```python
import math
import argparse # 导入argparse模块
# 用来装载参数的容器
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 给这个解析对象添加命令行参数
parser.add_argument('radius', type=int, help='Radius of cylinder')
parser.add_argument('height', type=int, help='Height of cylinder')
args = parser.parse_args() # 获取所有参数
def cylinder_volume(radius, height):
    vol = (math.pi) * (radius**2) * (height)
    return vol
if __name__ '__main__':
    print(cylinder_volume(args.radius, args.height))

```

1)首先导入argparse模块import argparse

2)创建一个 ArgumentParser 对象，该对象包含将命令行输入内容解析成 Python 数据的过程所需的全部功能。description是该对象的描述信息，可使用python test_argparse.py -h命令查看
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')

3)添加需要输入的命令行参数
parser.add_argument('radius', type=int, help='Radius of cylinder')
（）中依次为参数名；参数类型，声明这个参数的数据类型为int为了参与运算，默认数据类型为str；描述信息

4)args = parser.parse_args() ArgumentParser 通过 parse_args() 方法解析参数，获取到命令行中输入的参数。

5)将获取到的半径和高度args.radius, args.height作为参数传到方法中得出结果

## 7.16 -h参数查看描述信息

![image-20230914163023879]({{ '/assets/learning/01_python/image-20230914163023879.png' | relative_url }})

## 7.17 选择型参数

如果想改变输入的顺序或在输入参数同时携带参数名，可以使用选择型参数，在添加参数时参数名前加两个"-"。default设置默认值

```python
parser.add_argument('--radius', type=int, default=2, help='Radius of cylinder')
```

还有一种方法,通过“-”加上参数别名的形式，注意被"–"修饰的参数名必须存在：

```python
parser.add_argument('-r', '--radius', type=int, help='Radius of cylinder')
parser.add_argument('-H', '--height', type=int, help='Height of cylinder')
```

## 7.18 metavar

```python
parser.add_argument('-r', '--radius', type=int, metavar='', help='Radius of cylinder')
parser.add_argument('-H', '--height', type=int, metavar='', help='Height of cylinder')

```

## 7.19 required

required属性要求该参数必须被赋值，否则报错

```python
parser.add_argument('-r', '--radius', type=int, metavar='', required=True, help='Radius of cylinder')
parser.add_argument('-H', '--height', type=int, metavar='', required=True, help='Height of cylinder')

```

## 7.20 dest属性

可以通过dest=xxx来设置参数的变量名取代“–xxx”，同样地在代码中用args.xxx来获取参数的值。

```python
parser.add_argument('-r', dest='radius', type=int, help='Radius of cylinder')
parser.add_argument('-H', dest='height', type=int, help='Height of cylinder')

```

## 7.21 ArgumentTypeError

argparse中的异常类

```python
raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")

```

## 7.22 相关装饰器

### 7.22.1 @lru_cache

在计算机软件领域，缓存（Cache）指的是将部分数据存储在内存中，以便下次能够更快地访问这些数据，这也是一个典型的用空间换时间的例子。一般用于缓存的内存空间是固定的，当有更多的数据需要缓存的时候，需要将已缓存的部分数据清除后再将新的缓存数据放进去。需要清除哪些数据，就涉及到了缓存置换的策略，LRU（Least Recently Used，最近最少使用）是很常见的一个，也是 Python 中提供的缓存置换策略。

```python
import functools
# 注意 lru_cache 后的一对括号，证明这是带参数的装饰器
@functools.lru_cache()
def factorial(n):
 print(f"计算 {n} 的阶乘")
 return 1 if n <= 1 else n * factorial(n - 1)

'''计算 5 的阶乘
计算 4 的阶乘
计算 3 的阶乘
计算 2 的阶乘
计算 1 的阶乘
5! = 120
3! = 6'''
```

可以看到，这次在调用 `factorial(3)` 的时候没有打印相应的输出，也就是说 `factorial(3)` 是直接从缓存读取的结果，证明缓存生效了。

被 lru_cache 修饰的函数在被相同参数调用的时候，后续的调用都是直接从缓存读结果，而不用真正执行函数。

### 7.22.2 @memoized_parameterless_method

`@memoized_parameterless_method` 装饰器的作用是为一个无参数方法添加记忆功能。记忆（memoization）是一种优化技术，用于缓存函数或方法的结果，以避免重复计算。当相同的参数被传递给被装饰的方法时，装饰器会检查缓存中是否已经有了该参数的结果。如果有，则直接返回缓存中的结果，而不执行方法的计算过程。如果缓存中没有结果，则执行方法的计算过程，并将结果存储在缓存中供以后使用。

```python
def memoized_parameterless_method(method):
    cache = {}

    def wrapper(self):
        if method.__name__ not in cache:
            cache[method.__name__] = method(self)
            return cache[method.__name__]
        return wrapper

class MyClass:
    @memoized_parameterless_method
    def calculate(self):
        print("Calculating...")
        return 42

    obj = MyClass()
    print(obj.calculate()) # 第一次调用，会执行计算过程，并将结果存储在缓存中
    print(obj.calculate()) # 第二次调用，直接从缓存中返回结果，而不执行计算过程

```

在上面的示例中，`calculate` 方法被 `@memoized_parameterless_method` 装饰器修饰。第一次调用 `obj.calculate()` 时，会执行计算过程并将结果存储在缓存中。第二次调用时，直接从缓存中返回结果，而不再执行计算过程。

## 7.23 `filters = defaultdict(lambda:set())`

`filters` 是一个默认字典，它的默认值是一个空的集合（set）。这意味着当你查询一个不存在的键时，`filters` 会返回一个空的集合，而不是抛出 KeyError 异常。这样可以方便地向 `filters` 中添加新的键值对，并且不需要担心键是否存在的问题。

例如，当你执行 `filters['some_key'].add('some_value')` 时，如果 `some_key` 不存在，`filters` 会自动创建一个空的集合作为默认值，并向该集合中添加 `'some_value'`。如果 `some_key` 存在，它会将 `'some_value'` 添加到对应的集合中。

```python
from collections import defaultdict

# 创建默认字典
filters = defaultdict(lambda: set())
filters1 = {}
# 向默认字典中添加元素
filters['fruit'].add('apple')
filters['fruit'].add('banana')
filters['color'].add('red')
filters['color'].add('blue')

# filters1['fruit'] = set('apple')
filters1['fruit'] = {'apple'}
filters1['fruit'].add('banana')
filters1['color'] = 'red'
filters1['color'] = 'blue'

print(filters)
print(filters1)
```

输出:

```python
defaultdict(<function <lambda> at 0x0000015BC12AF4C0>, {'fruit': {'banana', 'apple'}, 'color': {'red', 'blue'}})
{'fruit': {'banana', 'apple'}, 'color': 'blue'}
```

## 7.24 csr_matrix

`csr_matrix` 是 SciPy 库中的一个函数，用于创建 Compressed Sparse Row (CSR) 格式的稀疏矩阵。稀疏矩阵是指大部分元素为零的矩阵，而 CSR 格式是一种用于高效存储和处理稀疏矩阵的数据结构。

`csr_matrix` 函数的语法是：

```python
scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(M, N))
```

其中：

- `data` 是一个一维数组，包含非零元素的值。
- `row_ind` 是一个一维数组，包含每个非零元素所在的行索引。
- `col_ind` 是一个一维数组，包含每个非零元素所在的列索引。
- `shape` 是一个元组，表示稀疏矩阵的形状。

例如，我们可以使用 `csr_matrix` 函数来创建一个 CSR 格式的稀疏矩阵：

```python
import scipy.sparse

data = [3, 1, 2, 4] # 非零元素的值
row_ind = [0, 0, 1, 2] # 非零元素的行索引
col_ind = [0, 2, 2, 0] # 非零元素的列索引

csr_mat = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(3, 3))
print(csr_mat)
```

这将创建一个 3x3 的稀疏矩阵，其中非零元素的值和位置由 `data`、`row_ind` 和 `col_ind` 指定。这种格式的稀疏矩阵在存储和处理大型稀疏矩阵时非常高效。

## 7.25 sorted()

在Python中，sorted()函数用于对可迭代对象（如列表、元组、字典等）进行排序。它的基本语法如下：

```python
sorted(iterable, key=None, reverse=False)
```

参数说明：

- iterable: 要排序的可迭代对象，如列表、元组等。
- key: 用于指定一个函数，该函数将作为排序的关键字，可以根据该函数返回的结果对可迭代对象进行排序。默认值为None，表示直接对可迭代对象中的元素进行排序。
- reverse: 用于指定排序顺序，True表示降序，False表示升序。默认值为False。

示例：

```python
# 对列表进行排序
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
sorted_list = sorted(my_list)
print(sorted_list) # 输出：[1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

# 对元组进行排序
my_tuple = (3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5)
sorted_tuple = sorted(my_tuple)
print(sorted_tuple) # 输出：[1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

# 对字典进行排序.
my_dict = {'c': 3, 'a': 1, 'b': 2}
sorted_dict = sorted(my_dict.items(), key=lambda x: x[1])
print(sorted_dict) # 输出：[('a', 1), ('b', 2), ('c', 3)]
```

对字典使用sorted()函数时，默认情况下会按照字典的键（key）进行排序。这意味着sorted()函数会根据字典的键对键值对进行排序，并返回一个按照键排序的键值对列表。

21 np.random.choice

`np.random.choice(a[,size,replace,p,axis])`函数有几个常用的参数，下面是这些参数的说明：

1. `a`：数组或整数。如果`a`是一个数组，则从该数组中进行随机选择。如果`a`是一个整数，那么随机选择的范围是从0到`a-1`。
2. `size`：整数或整数元组。指定要抽取的样本数量。如果是整数，则表示抽取的样本数量；如果是整数元组，如`(m, n)`，则表示抽取一个`m*n`的数组，其中包含从`a`中随机选择的样本。
3. `replace`：布尔值。指定是否允许重复选择。如果为`False`，则不允许重复选择；如果为`True`，则允许重复选择。
4. `p`：一维数组，元素为非负数字，总和为1。指定每个元素被选择的概率。如果不提供`p`，则所有元素被选择的概率相等。
5. `axis`：整数。如果`a`是多维数组，则指定沿着哪个轴进行抽样。默认为0，表示沿着第一个轴进行抽样。

## 7.26 tertools.groupby

itertools.groupby是Python标准库中的一个函数，它用于对可迭代对象进行分组。该函数接受一个可迭代对象和一个可选的key函数作为参数，然后返回一个迭代器，该迭代器生成由key函数返回值相同的元素所组成的子序列。

具体来说，itertools.groupby会将可迭代对象中连续且相同的元素分为一组，并返回一个由(key, group)组成的元组，其中key是分组的依据，group是对应的子序列。

例如，对于可迭代对象[[1], [1], 2, 2, 3, 3, 3, 4, 4, 4, 4]，使用itertools.groupby函数可以将其分组为[([1], [[1], 1]]), (2, [2, 2]), (3, [3, 3, 3]), (4, [4, 4, 4, 4])]。在这个例子中，相同的数字被分为一组，并且以数字本身作为分组的依据。其中，每个原色可以为列表等复杂元素

itertools.groupby函数在处理需要对可迭代对象进行分组的情况下非常有用，可以帮助简化代码并提高效率。

```python
import itertools
all_bodies = [[1,2,3], [1,2,3], [2,3,4], [2,3,4],[2,3,4], [4, 4, 4]]
all_bodies.sort() #先排序
unique_bodies = list(x for x, _ in itertools.groupby(all_bodies)) #去重
print(unique_bodies)
```

## 7.28 Counter()

ounter函数是Python中collections模块中的一个类，用于对可迭代对象中元素的出现次数进行计数。它会返回一个字典，其中包含了每个元素作为键，以及该元素出现的次数作为值。可以用它来方便地统计列表、元组、字符串等数据结构中元素的出现次数。例如，Counter(["a", "b", "a", "c", "a"])会返回一个字典，其中"a"出现了3次，"b"出现了1次，"c"出现了1次。

```python
lengths = []
for rel in rules_dict:
    lengths += [len(x["body_rels"]) for x in rules_dict[rel]]
    rule_lengths = [(k, v) for k, v in Counter(lengths).items()]
    print("Number of rules by length: ", sorted(rule_lengths))

    '''Number of rules by length: [(1, 7847), (2, 7328), (3, 11706)]'''
```

## 7.29 itertools.islice

tertools.islice是Python中的一个函数，用于从可迭代对象中按照指定的索引范围进行切片操作。它的语法如下：

itertools.islice(iterable, start, stop, step)

其中，iterable是要进行切片操作的可迭代对象，start是切片的起始索引，stop是切片的结束索引，step是切片的步长。

itertools.islice返回一个迭代器，该迭代器会按照指定的索引范围对可迭代对象进行切片操作，并返回切片后的元素序列。

例如，可以使用itertools.islice来对列表、元组、字符串等可迭代对象进行切片操作，从而获取指定范围内的元素。例如：

```python
import itertools
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# 从索引2到索引7，步长为2进行切片
result = itertools.islice(data, 2, 7, 2)
for item in result:
    print(item)
```

以上代码会输出切片后的元素序列：3, 5, 7。

## 7.30 np.where

`np.where`是numpy库中的一个函数，它用于根据指定的条件返回数组中满足条件的元素的索引。`np.where`的基本语法是`np.where(condition, x, y)`，其中condition是一个布尔型数组或条件表达式，x和y是对应位置上的元素，当condition中的元素为True时，返回x中对应位置上的元素，否则返回y中对应位置上的元素。

例如，`np.where(arr > 0, 1, -1)`会返回一个新的数组，其中arr中大于0的元素用1填充，小于等于0的元素用-1填充。

```python
import numpy as np
arr = np.array([1, -2, 3, -4, 5])
result = np.where(arr > 0, 1, -1)
print(result)
# [ 1 -1 1 -1 1]
```

`np.where`返回一个长度为2的tuple，每个元素都是一个一维数组。第一个数组表示满足条件的元素在第一个维度（行）上的索引，第二个数组表示满足条件的元素在第二个维度（列）上的索引。
