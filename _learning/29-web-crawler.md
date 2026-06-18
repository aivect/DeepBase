---
layout: article
title: "网络爬虫基础"
key: learning-29-web-crawler
permalink: /learning/29-web-crawler.html
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

# 1 基础

通用爬虫基础流程：发送网络请求获取页面数据 → 清洗、解析提取目标信息 → 数据去重与存储持久化。

## 1.1 web请求过程剖析

**1、服务器渲染：**

我们在请求到服务器的时候, 服务器直接把数据全部写⼊到html中, 我们浏览器就能直接拿到带有数据的html内容

**2、客户端渲染:**

第一次请求服务器只返回一个html骨架，第二次请求拿到数据，进行数据展示在页面源代码中，看不到数据。（浏览器检查XHR第二次请求中可以找到）

## 1.2 http协议

协议: 就是两个计算机之间为了能够流畅的进⾏沟通⽽设置的⼀个君 ⼦协定.

HTTP协议, Hyper Text Transfer Protocol（超⽂本传输协议）的缩 写,是⽤于从万维⽹（WWW:World Wide Web ）服务器传输超⽂本 到本地浏览器的传送协议. 直⽩点⼉, 就是浏览器和服务器之间的数 据交互遵守的就是HTTP协议.

HTTP协议把⼀条消息分为三⼤块内容. ⽆论是请求还是响应都是三块内容

1、请求:

1. 请求⾏ -> 请求⽅式(get/post) **请求url地址** 协议
2. 请求头 -> 放⼀些服务器要使⽤的附加信息
3. 请求体 -> ⼀般放⼀些请求参数

2、 响应:

1. 状态⾏ -> 协议状态码
2. 响应头 -> 放⼀些客户端要使⽤的⼀些附加信息
3. 响应体 -> 服务器返回的真正客户端要⽤的内容(HTML、Json)等

3、请求头中最常⻅的⼀些重要内容(爬⾍需要):

1. **User-Agent** : 请求载体的身份标识(⽤啥发送的请求，当不添加user-agent默认为python-requests（通过打印resp.requests.headers查看），因此需要添加浏览器user-agent，否则获取不到数据)
2. **Referer**: 防盗链(这次请求是从哪个⻚⾯来的? 反爬会⽤到)
3. **cookie**: 本地字符串数据信息(⽤户登录信息, 反爬的token)

4、 响应头中⼀些重要的内容:

1. cookie: 本地字符串数据信息(⽤户登录信息, 反爬的token)
2. 各种神奇的莫名其妙的字符串(这个需要经验了, ⼀般都是token 字样, 防⽌各种攻击和反爬)

5、请求⽅式:

GET: 显示提交。查询数据用的多

POST: 隐示提交

## 1.3 request

```python
# 案例1. 抓取搜狗搜索内容
kw = input("请输⼊你要搜索的内容:")
response =requests.get(f"https://www.sogou.com/web?query={kw}") # 发送get请求
# print(response.text) # 直接拿结果(⽂本)
with open("sogou.html", mode="w", encoding="utf8") as f:
f.write(response.text)
responce.close()
```

```python
#案例2，抓取豆瓣电影
import requests
url = 'https://movie.douban.com/j/chart/top_list'
#封装参数
param = {
 'type':24,
 'interval_id':'100:90',
 'action':"",
 'start':0,
 'limit':20,
}
headers = {
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36'
}
resp = requests.get(url=url, params=param, headers=headers)
# print(resp.request.headers)
#转为字典格式
print(resp.json())
resp.close()
```

# 2 提取和解析数据

## 2.1 数据解析概述

在上⼀章中, 我们基本上掌握了抓取整个⽹⻚的基本技能. 但是呢, ⼤ 多数情况下, 我们并不需要整个⽹⻚的内容, 只是需要那么⼀⼩部分. 怎么办呢? 这就涉及到了数据提取的问题. 本课程中, 提供三种解析⽅式: 可使用混合

1. re解析 2. bs4解析 3. xpath解析

## 2.2 RE 正则表达式

Regular Expression, 正则表达式, ⼀种使⽤表达式的⽅式对字符串 进⾏匹配的语法规则. 我们抓取到的⽹⻚源代码本质上就是⼀个超⻓的字符串, 想从⾥⾯提 取内容.⽤正则再合适不过了。

正则的优点: 速度快, 效率⾼, 准确性⾼ 正则的缺点: 新⼿上⼿难度有 点⼉⾼. 不过只要掌握了正则编写的逻辑关系, 写出⼀个提取⻚⾯内容的正则 其实并不复杂。

正则的语法: 使⽤元字符进⾏排列组合⽤来匹配字符串 在线测试正 则表达式https://tool.oschina.net/regex/

2.1、**`元字符: 具有固定含义的特殊符号。常⽤元字符:`**

```python
1. .匹配除换⾏符以外的任意字符(**)
2. \w 匹配字⺟或数字或下划线 (**)
3. \s 匹配任意的空⽩符
4. \d 匹配数字 (**)
5. \n 匹配⼀个换⾏符
6. \t 匹配⼀个制表符

7. ^ 匹配字符串的开始
8. $ 匹配字符串的结尾

9. \W 匹配⾮字⺟或数字或下划线
10. \D 匹配⾮数字
11. \S 匹配⾮空⽩符
12. a|b 匹配字符a或字符b
13. () 匹配括号内的表达式，也表示⼀个组
14. [...] 匹配字符组中的字符 [a-zA-Z0-9],-是至(**)
15. [^...] 匹配除了字符组中字符的所有字符
```

2.2、`量词: 控制 前⾯ 的元字符出现的次数`

```python
1. *重复零次或更多次 (**)
2. +重复⼀次或更多次： \d+ 重复多个数字(**)
3. ? 重复零次或⼀次,尽可能重复前面的次数少 (**)
4. {n} 重复n次
5. {n,} 重复n次或更多次
6. {n,m} 重复n到m次
```

2.3、`贪婪匹配和惰性匹配`

```python
1. .* 贪婪匹配 #
2. .*? 惰性匹配 #尽可能少的匹配内容。
	案例1：
	str: sffsdfzdvdfbz
	reg: .*?z(每次匹配以z结尾)
	共找到 2 处匹配：
		sffsdfz
		dvdfbz
	案例2
	str: 玩⼉吃鸡游戏, 晚上⼀起上游戏, ⼲嘛呢? 打游戏啊
	reg: 玩⼉.*?游戏 (先找字符串最远的，再通过回溯找出最近的，返回)
	此时匹配的是: 玩⼉吃鸡游戏
```

### r 的作用是去除转义字符.

### 以 f开头表示在字符串内支持大括号内的python 表达式

## 2.3 re模块

1、findall(正则表达式，待查找内容) 查找所有。返回list

```python
lst = re.findall("m", "mai le fo len, mai nimei!")
print(lst)
#输出 ['m', 'm', 'm']
```

2、 **findite**r, 和findall差不多. 只不过这时返回的是**迭代器**(重点)

```python
lst = re.finditer(r"\d+", "5点之前. 你要给我5000万")
for i in lst:
	print(i.group())
#输出
5
5000
```

3、 search 会进⾏匹配. 但是如果匹配到了第⼀个结果. 就会返回这 个结果. 如果匹配不上search返回的则是None

4、 match 只能从字符串的开头进⾏匹配 ,开始头不满足，则报错，即只匹配开头

```python
ret = re.match('a', 'abc').group()
print(ret) # a
```

5、 **compile()** 可以将⼀个⻓⻓的**正则**进⾏**预加载**. ⽅便后⾯的使⽤

```python
obj = re.compile(r"\d+")
ret = obj.finditer("我的电话号码是1008611，我女朋友的是10010")
for it in ret:
 print(it.group())
```

6、**(?P<分组名字>正则表达式)** 可以单独从匹配的内容中进一步筛选（注意加**小括号**）

```python
#"""
 """:用于表示多行字符串变量

s = """
<div class='1'><span id='10011'>中国联通</span></div>
<div class='1'><span id='10010'>中国电信</span></div>
<div class='1'><span id='10086'>中国移动</span></div>
"""
obj = re.compile(r"<div class='.*?'><span id='(?P<id>\d+)'>(?P<wawa>.*?)</span>", re.S)
					#re.S 是.能匹配换行符
					#r :原生字符，防止字符转义
re = obj.finditer(s)
for it in re:
 print(it.group("id"),end=" ")
 print(it.group("wawa"))
#输出
10011 中国联通
10010 中国电信
10086 中国移动
```

7、re.sub(**pattern**, **repl**, **string,** count=0, flags=0)

1）pattern：该参数表示**正则中的模式字符串**；
（2）repl：该参数表示**要替换的字符串**（即匹配到pattern后替换为repl），也**可以是个函数**；
（3）string：该参数表示要被处理（查找替换）的**原始字符串**；
（4）count：可选参数，表示是要替换的最大次数，而且必须是非负整数，该参数默认为0，即所有的匹配都会被替换；
（5）flags：可选参数，表示编译时用的匹配模式（如忽略大小写、多行模式等），数字形式，默认为0。

## 2.4 项目1 豆瓣top250

```python
#1、拿到页面源代码
#2、通过re来提取想要的有效内容
#拿到页面源代码
#通过re来提取想要的有效内容
import requests
import re
import csv
url = 'https://movie.douban.com/top250'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36'
}
resp = requests.get(url,headers=headers)
page_context = resp.text
#解析数据
obj = re.compile(r'<li>.*?<div class="item">.*?<em class="">(?P<rank>.*?)</em>.*?'
                 r'<span class="title">(?P<title>.*?)'
                 r'</span>.*?<p class="">.*?<br>(?P<year>.*?)&nbsp.*?'
                 r'<span class="rating_num" property="v:average">(?P<score>.*?)</span>.*?'
                 r'<span>(?P<num>.*?)人评价</span>',re.S)
#开始匹配
result = obj.finditer(page_context)
#写入文件
f = open("data_boudan.csv",mode="w",encoding="utf-8")#
#通过csv向f写
csvwriter = csv.writer(f)
for it in result:
    # print(it.group("rank"))
    # print(it.group("title"),end=",")
    # print(it.group("year").strip(),end=" ")
    # print(it.group("score"),end=" ")
    # print(it.group("num"))
    #转为字典
    dic = it.groupdict()
    dic['year'] = dic['year'].strip()
    csvwriter.writerow(dic.values())
    f.close()
    print("over")
```

## 2.5 手刃电影天堂

```python
#1.定位到2021必看片
#2.从2021必看片中提取到子叶面的连接地址
#3、请求子页面的地址，拿到我们想要的下载地址
import requests
import re
domain = 'https://www.dy2018.com/'
#verify=False 去掉安全验证
resp = requests.get(domain,verify=False)
#默认解码是utf-8
resp.encoding = 'gb2312'
#1.定位到2021必看片
obj1 = re.compile(r'2022必看热片.*?<ul>(?P<ul>.*?)</ul>',re.S)
#获取子链接
obj2 = re.compile(r"<a href='(?P<href>.*?)'",re.S)
#进入子连接获取详细内容
obj3 = re.compile(r'<br />◎片　　名(?P<movie>.*?)<br />.*?'
                  r'<td style="WORD-WRAP: break-word" bgcolor="#fdfddf"><a href="(?P<download>.*?)">',re.S)
result1 = obj1.finditer(resp.text)

child_href_list = []
for it1 in result1:
    ul = it1.group('ul')
    # 2.提取子叶面连接地址
    result2 = obj2.finditer(ul)
    for it2 in result2:
        #获取子页面连接
        child_href = domain + it2.group('href').strip()
        #3、把子页面连接保存到队列中
        child_href_list.append(child_href)
        # 3、提取页面内容
        for href in child_href_list:
            child_resp = requests.get(href,verify=False)
            child_resp.encoding = 'gb2312'
            result3 = obj3.search(child_resp.text)
            print(result3.group('movie').strip())
            print(result3.group('download'))
```

## 2.6 xpath解析

XPath是⼀⻔在 XML ( Extensible Markup Language，可扩展标记语言，可自定义标记)⽂档中查找信息的语⾔. XPath可⽤来在 XML ⽂档中对元素和属性进⾏遍历. ⽽我们熟知的HTML恰巧属于XML的 ⼀个⼦集. 所以完全可以⽤xpath去查找html中的内容.

HTML当作XML定义出来的特定用途（表示网页）的标记语言

/text()获取标记中的文本

**//**nick 子 **孙** 节点的nick节点都获取到

*任意的节点，通配符

例子1

```python
from lxml import etree
xml = """
<book>
 <id>1</id>
 <name>野花遍地⾹</name>
 <price>1.23</price>
 <nick>臭⾖腐</nick>
 <author>
 <nick id="10086">周⼤强</nick>
 <nick id="10010">周芷若</nick>
 <nick class="joy">周杰伦</nick>
 <nick class="jolin">蔡依林</nick>
 <div>
 <nick>惹了</nick>
 </div>
 </author>
 <partner>
 <nick id="ppc">胖胖陈</nick>
 <nick id="ppbc">胖胖不陈</nick>
 </partner>
</book>
"""
# 解析html字符串
tree = etree.XML(xml)
# text()拿文本
#result = tree.xpath("/book/name/text()")
# //nick:子孙节点的nick都获取到
#result = tree.xpath("/book/author//nick/text()")
#result = tree.xpath("/book//nick/text()")
# * 任意的节点，通配符
result = tree.xpath("/book/author/*/nick/text()")
print(result)
```

例子2

```python
#解析html文件,默认解析器为xml
#etree.parse("文件名")
#tree = etree.parse("1.html")
#默认是XML解析器，碰到不规范的html文件时就会解析错误，增加解析器
parser = etree.HTMLParser(encoding="utf-8")
tree = etree.parse("1.html",parser=parser)

#1、获取ul下所有li下的内容
result_1 = tree.xpath("/html/body/ul/li/a/text()")

#2、li[1] 获取第一个
result_2 = tree.xpath("/html/body/ul/li[1]/a/text()")

#3、通过@获取ul下所有a标签href属性的值
result_3 = tree.xpath("/html/body/ul/li/a/@href")
# result_3 = tree.xpath("/html/body/div/@class")
# result_3 = tree.xpath("/html/body/div/text()")

#4、a[@href='dapao'] 获取a标签下的属性href = ‘dapao’
result_4 = tree.xpath("/html/body/ol/li/a[@href='dapao']/text()")
print(result_1)

ol_li_list = tree.xpath("/html/body/ol/li")
for li in ol_li_list:
    #在li中继续相对寻找
    result = li.xpath("./a/text()")
    print(result)
    #拿取a标签下属性href的值
    result2 = li.xpath("./a/@href")
    print(result2)
```

## 2.7 实战抓取猪八戒网信息

```python
#1、拿到页面源代码
#2、提取和解析数据
import requests
from lxml import etree
url = "https://sichuan.zbj.com/search/f/?kw=大数据"

#获取源代码
resp = requests.get(url)

#加载解析html源码
html = etree.HTML(resp.text)
divs = html.xpath("/html/body/div[6]/div/div/div[2]/div[5]/div[1]/div")

#相对解析
for div in divs:
    store_name = div.xpath('./div/div/a[1]/div[1]/p/text()')[1].strip("\n")
    adress = div.xpath('./div/div/a[1]/div[1]/div/span/text()')[0]#列表变为字符串
    price = div.xpath('./div/div/a[2]/div[2]/div[1]/span[1]/text()')[0].strip("¥")
    #"大数据".join()用大数据连接列表中的元素
    title = "大数据".join(div.xpath('./div/div/a[2]/div[2]/div[2]/p/text()'))
    print(store_name)
```

# 3 requests进阶

## 3.1 概述

header为HTTP协 议中的请求头. ⼀般存放⼀些和请求内容⽆关的数据. 有时也会存放 ⼀些安全验证信息.⽐如常⻅的User-Agent, token, cookie等.

本章内容: 1. 模拟浏览器登录->处理cookie 2. 防盗链处理-> 抓取梨视频数据 3. 代理 -> 防⽌被封IP

## 3.2 爬取小说网书架信息(登录机制session)

```python
#登陆，得到cookie
#带着cookie，去请求得到书架url,得到数据内容
'''可以通过session进行请求——>session可认为一连串的请求，
整个过程，cookie不会丢失'''
import requests

#session会话
session = requests.session()
# session.encoding = "utf-8"
data = {
    "loginName": "18614075987",
    "password": "q6035945"
}

#1、模拟登录
url = "https://passport.17k.com/ck/user/login"
#登录cookie存入session,用于方法1
session.post(url,data=data)
url_shujia = 'https://user.17k.com/ck/author/shelf?page=1&appKey=2406394919'
#方法1
resp = session.get(url_shujia)
#方法2
# resp = requests.get(url_shujia,headers={"Cookie": "………………"})
print(resp.text)
```

## 3.3 反扒机制防盗链: 抓取梨视频

```python
#1、拿到contId
#2、拿到video中的srcUrl
#3、对srcUrl修改
#4、下载
import requests
import re
url = "https://www.pearvideo.com/video_1728850"
contId = url.split("_")[1]
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36",
    #防盗链：溯源。对所发请求溯源，看是否是符合设置的请求。如1->2->3,不能跳过2或跳过1,、2访问3。
    "Referer": url
}
videoStatusUrl = f"https://www.pearvideo.com/videoStatus.jsp?contId={contId}&mrd=0.3389920915159985"
resp = requests.get(videoStatusUrl,headers=headers)
#resp.json()的爬取数据转为字典
dic = resp.json()

srcUrl = dic["videoInfo"]["videos"]["srcUrl"]
systemTime = dic["systemTime"]
srcUrl = srcUrl.replace(systemTime,f"cont-"+contId)
#下载视频
with open("a.mp4",mode="wb") as f:
    f.write(requests.get(srcUrl).content)
    print("over!")
```

## 3.4 反扒机制代理

**原理：通过一个第三方的及其去发送请求**

```python
import requests
#代理
proxies = {
 "https":"https://218.60.8.83:3129"
}
resp = requests.get("https://www.baidu.com",proxies=proxies)
resp.encoding = "utf-8"
print(resp.text)
```

# 4 爬虫效率

## 4.1 梗概

选择多线程, 多进程, 协程 等操作完异步爬⾍.

## 4.2 多线程

```python
#导入线程类
from threading import Thread
#方法1
def func():
    for i in range(100):
    print("func",i)
#__name__:内置变量
#if __name__"__main__":启动代码块。
# 如果这个文件是作为模块被其他文件调用，不会执行这里面的代码。 只有执行这个文件时， if里面的语句才会被执行
if __name__"__main__":
    #主线程产生子线程，并安排子线程执行任务为func()函数
    # 线程1,实参传递参数必须是元组，一个参数后必须加“，”,否则为字符串
    t1 = Thread(target=func,args=("张三",))
    t1.start()
    t2 = Thread(target=func,args=("李四",))#线程2
    t2.start()
    #线程进入工作就绪态

    for i in range(100):
    print("main",i)

#方法2
class MyThread(Thread):
     def run(self):#必须写，当线程执行时，被执行的就是run()
         for i in range(100):
         print("子线程",i)
if __name__ '__main__':
     t = MyThread()
     #开启线程
     t.start()
     for i in range(50):
     print("主线程",i)
```

## 4.3 多进程

```python
#导入进程类
from multiprocessing import Process
def func():
    for i in range(100):
        print("次进程",i)
        if __name__ '__main__':
            p = Process(target=func())
            #开启线程
            p.start()
            for i in range(50):
                print("主进程",i)
```

## 4.4 线程池和进程池

创建⼀个⼤池⼦, 存放固定数量的线程. 然后把我们要执⾏的任务丢 给线程池. 由线程池去分配哪个线程来完成该任务. 其他的事情都不 需要你来管 。

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# 线程池
def fn(name):
    for i in range(10):
        print(name, i)

if __name__ '__main__':
    #创建进程池
    #with ThreadPoolExecutor(10) as t:
    #创建线程池
    with ThreadPoolExecutor(10) as t:
        #100个任务交给线程池
        for i in range(100):
            #线程池中各线程执行fn(name),参数name
            t.submit(fn, name=f"线程{i}")
            #等待线程池中的任务全部执行完后，才继续执行(守护)
            print("over")
```

## 4.5 线程池_抓取新发地

```python
import requests
from lxml import etree
import csv
from concurrent.futures import ThreadPoolExecutor
f = open("data1.csv",mode="w",encoding="utf-8")
csvwriter = csv.writer(f)

def download_one_page(url):
     #拿到页面源代码
     resp = requests.get(url)
     html = etree.HTML(resp.text)
     table = html.xpath("/html/body/div[2]/div/div/div/div[4]/div[1]/div/table")[0]
     #对数据简单处理：\\ /去掉，由第一行是列名，所以要从第二行开始获取数据
     #方法1 trs = table.xpath("./tr")[1:]
     #方法2
     trs = table.xpath("./tr[position()>1]")
     #拿到每个tr
     for tr in trs:
     	txt = tr.xpath("./td/text()")
         #生成器效率更高
         txt = (item.replace("\\","").replace("/","") for item in txt)
         #print(list(txt))
         csvwriter.writerow(txt)

if __name__ '__main__':
    #创建线程池
    with ThreadPoolExecutor(50) as t:
    for i in range(1,200): 		 t.submit(download_one_page,"http://www.xinfadi.com.cn/marketanalysis/0/list/{i}.shtml")
    print("over")
```

## 4.6 协程

⼀个线程遇到了IO操 作, CPU就会⾃动的切换到其他线程进⾏执⾏。 当线程中遇 到了IO操作的时候, 将线程中的任务进⾏切换, 切换成⾮ IO操作. 等 原来的IO执⾏完了. 再恢复回原来的任务中。 该模型被称为**协程**。协程是由程序而不是操作系统来切换。

```python
import asyncio
#异步协程函数
async def func():
 print("我爱你")
 #当前线程阻塞

if __name__ '__main__':
 #函数执行得到异步协程对象
 g = func()
 asyncio
```

```python
import asyncio
import time
#异步协程函数
# await: 当该任务被挂起后,CPU会⾃动切换到其他任务中
async def func1():
    print("func1, start")
    #await挂起当前线程，实现异步操作
    await asyncio.sleep(3)
    print("func1, end")
async def func2():
    print("func2, start")
    #异步协程完成时间为三者最多的4s
    await asyncio.sleep(4)
    print("func2, end")
async def func3():
    print("func3, start")
    await asyncio.sleep(2)
    print("func3, end")

async def main():
    #await挂起操作放在协程对象前
    # 第一种写法
    # f1 = func1()
    # await f1
    #第二种写法
    tasks = [ # 协程任务列表
        asyncio.create_task(func1()), # 创建协程任务
        asyncio.create_task(func2()),
        asyncio.create_task(func3())
    ]
    await asyncio.wait(tasks)

if __name__ '__main__':
    t1 = time.time()
    asyncio.run(main())
    t2 = time.time()
    print(t2-t1)
```

```python
import asyncio
import time
async def download(url):
    print("开始抓取")
    await asyncio.sleep(2) # 网络请求，异步操作
    print("下载结束", url)
    return "⽼⼦是源码你信么"
async def main():
    urls = [
        "http://www.baidu.com",
        "http://www.h.com",
        "http://luoyonghao.com"
    ]
    # ⽣成任务列表
    tasks = []
    for url in urls:
        d = asyncio.create_task(download(url))
        tasks.append(d)
        await asyncio.wait(tasks)
        # 方法2
        # tasks = [asyncio.create_task(download(url)) for url in urls]
        # await asyncio.wait(tasks)
if __name__ '__main__':
    t1 = time.time()
    asyncio.run(main())
    t2 = time.time()
    print(t2-t1)
```

# 5 其他模块

## 5.1 selenium 基于浏览器自动化的一个模块

selenium模块和爬虫之间具有怎样的关联?
-便捷的获取网站中动态加载的数据
-便捷实现模拟登录
se len ium使用流程:
环境安装: pip install se Lenium
-下载一个浏览器的驱动程序

```python
from selenium import webdriver

chromedriver = r'D:\chromedriver\chromedriver.exe'
driver = webdriver.Chrome(executable_path=chromedriver)
#发起请求
driver.get('https://mall.jd.com/view_search-
394872-0-0-1-24-2.html')
# 获取加载源码
resp = driver.page_source
#标签定位
search_innput = driver.find_element_by_id('q')
#标签交互 搜索框输入信息
search_innput.send_keys('榴莲')
#点击搜索按钮
#('.btn-search'中.表示是寻找类class='btn-search')
btn = driver.find_element_by_css_selector('.btn-search')
btn.click()
#执行js程序 window.scrollTo(x,y)，横向与纵向滚动多少
driver.execute_script('window.scrollTo(0,document.body.scrollHeight);')
#后退
driver.back()
#前进
driver.forward()
#关闭浏览器
driver.quit()
#刷新
dirver.refresh()

```

# 附录：

爬虫 AI 工具： [crawl4ai](https://github.com/unclecode/crawl4ai)（As a contributor）
