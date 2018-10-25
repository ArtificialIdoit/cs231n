"""
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[int(len(arr)/2)]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle +quicksort(right)
print(quicksort([3,6,8,10,1,2,1]))
"""

"""
x = 3
print(x)
print(type(x))
print(x+1)
print(x-1)
print(x*2)
print(x**2)
x += 1
print(x)
x *= 2
print(x)
y = 2.5
print(type(y))
print(y,y+1,y-1,y*2,y**2)
"""

"""
hello = 'hello'
world = "world"
print(hello)
print(len(hello))
hw = hello + ' ' +world
print(hw)
hw12 = '%s %s %d' %(hello,world,12)
print(type(hw12))
print(hw12)
"""

"""
s = 'hello'
print(s.capitalize())
print(s.upper())
print(s.rjust(7))
print(s.center(8)) #奇数个，多一格往右边放
print(s.replace('l','ell'))
print('  world '.strip())
"""

"""
nums = range(5)
print(type(nums))
print(nums)
print(nums[2])
print(nums[2:4])
print(nums[2:])
print(nums[:4])
print(nums[:])
print(nums[:-1])
nums2[2:4] = [8,9]
print(nums,nums2)
"""

'''
animals = ['cat','dog','bird']
for animal in animals:
    print(animal)
for idx, animal in enumerate(animals):
    print(idx,animal)
    print('#%d: %s' % (idx + 1, animal))
'''

'''
nums = [0,1,2,3,4]
nums2 = [x**2 for x in nums] #if x > 0 会把0筛掉
print(nums2)

mcase = {'a': 10, 'b': 34}
wcase = {v:k for k,v in mcase.items()}
print(wcase)
'''

'''
#variable = [out_exp_res for out_exp in input_list if out_exp == 2]
  out_exp_res:　　列表生成元素表达式，可以是有返回值的函数。
  for out_exp in input_list：　　迭代input_list将out_exp传入out_exp_res表达式中。
  if out_exp == 2：　　根据条件过滤哪些值可以。
'''

'''
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat'])
print('cat' in d)
d['fish'] = 'wet'
print(d['fish'])
print(d.get('money','N/A'))
print(d.get('fish','N/A'))
del d['fish']
if 'money' in d:
    del d['money']
print(d.get('fish','N/A'))
'''

'''
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d :
    legs = d[animal]
    print('A %s has %d legs' %(animal,legs))
for animal,legs in d.items():
    print ('A %s has %d legs' %(animal,legs))
'''

'''
nums = [0,1,2,3,4]
even_nums_to_square = {x: x**2 for x in nums if x % 2 == 0}
print(even_nums_to_square)
'''

'''
animals = {'cat','dog'}
print('cat' in animals)
print('fish' in animals)
print(len(animals))
animals.add('fish')
print('fish' in animals)
print(len(animals))
animals.add('cat')
print(len(animals))
animals.remove('cat')
print(animals)
'''

'''
animals = ['dog','cat','fish']
for animal in animals:
    print(animal)
for idx,animal in enumerate(animals):
    print('#%d:%s' %(idx+1,animal))
'''

'''
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)
'''

'''
d = {(x,x+1): x for x in range(10)}
print(d)
t = (5,6)
print(type(t))
print(d[t])
print(d[(1,2)])
'''

'''
def sign(x):
    if x < 0:
        return 'negtive'
    elif x > 0 :
        return 'positive'
    else:
        return 'zero'
    
for x in [-1,0,1]:
    print(sign(x))
'''

'''
def hello(name,load = False):
    if load:
        return('HELLO,' + name.upper()+'!')
    else:
        return('Hello,' + name)

print(hello('Bob'))
print(hello('Fred', load = True))
'''

'''
class Greeter():
   def greet(self,name,loud = False): #必须加self
      if loud:
          print('HELLO,'+name.upper()+'!')
      else:
          print('Hello，'+name)
g = Greeter()
g.greet('Bob')
g.greet('Fred',loud = True)
'''

'''
import numpy as np
l = [1,2,3]
a = np.array(l)
l.append(4)
print(l)
print(type(a))
print(a.shape)
print(a)
a[0] = 5
print(a)
b = np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b)
'''

'''
import numpy as np
a = np.zeros((2,2),dtype = int)
print(a)
b = np.ones((2,2,2))
print(b)
c = np.full((1,3),7)
print(c)
d = np.eye(3)
print(d)
e = np.random.random((3,3))
print(e)
f = np.random.rand(3)
print(f)
g = np.random.randn(3)
print(f)
'''

'''
import numpy as np
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
b = a[:2,1:3]
print(b)
print(a[0,1])
b[0,0] = 77
print(b[0,0])
print(a[0,1])
'''

'''
import numpy as np
a = np.array([[1,2], [3, 4], [5, 6]])
print(a[[0,1,2],[0,1,0]]) #批量访问数组a中特定元素的方法
print(np.array([a[0,0],a[1,1],a[2,0]]))
print([a[0,0],a[1,1]])
print(np.array([a[0,1],a[0,1]]))
'''

'''
import numpy as np
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_1 = a[1,:]
row_2 = a[1:2,:]
row_1_reshape = np.reshape(row_1,[2,2])
print(row_1,row_1.shape)
print(row_2,row_2.shape)
row_1[0]+=10
row_2[0,0]+=10
print(a)
col_1 = a[:,1]
col_2 = a[:,1:2,]
print(col_1,col_1.shape)
print(col_2,col_2.shape)
'''

'''
import numpy as np
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
b = np.array([0,2,0,1]) 
c = np.arange(4) #本质上是一个ndarray
a[c,b] += 10
print(a)
'''

'''
import numpy as np
a = np.array([[1,2],[3,4],[5,6]])
bool_idx = (a>2)
print(bool_idx)
print(a[bool_idx])
print(a[a>2])
'''

'''
import numpy as np

x = np.array([1,2])
print(type(x))
print(x.dtype)

x = np.array([1.0,2.0])
print(x.dtype)

x = np.array([1.0,2.0],dtype = np.int64)
print(x.dtype)
print(x)
'''

'''
import numpy as np
x = np.array([[1,2],[3,4]],dtype = float)
y = np.array([[5,6],[7,8]],dtype = float)
print(x.dtype)
print(x+y)
print(np.add(x,y))
print(x*y)
print(np.multiply(x,y))
print(np.dot(x,y)) #矩阵乘法
print(x/y)
print(np.divide(x,y))
print(np.sqrt(x))
print(np.reshape(x,[1,4]))
'''

'''
import numpy as np
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11,12])

print(v.dot(w))
print(np.dot(v,w))

print(x.dot(v))
print(np.dot(x,v))

print(x.dot(y))
print(np.dot(x,y))
'''

'''
import numpy as np
x = np.array([[1,2],[3,4]])
print(np.sum(x))
print(x.sum(axis=0)) #合并行
print(np.sum(x,axis = 1)) #合并列
#两个处理结果的shape都是(2,)

print(x)
print(x.T)
v = np.array([[1,2,3]])
print(v)
print(v.T)
print(v.shape)
'''

'''
import numpy as np
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
y = np.empty_like(x)
v = np.array([[1, 0, 1]])
vv = np.tile(v,(4,1)) #不输入两个维度就是一维的
print(vv)
print(x+v)
print(x.dot(v.T))
'''

'''
import numpy as np
v = np.array([[1,2,3]])
w = np.array([[4,5]])
print(v.T*w)
print(v.reshape((3,1))*w)
x = np.array([[1,2,3], [4,5,6]])
print(x+v)
print((x.T+w).T)
print(x+w.T)
print(x*2)
'''

'''
from scipy.misc import imread,imsave,imresize
img = imread('cat.jpeg')
print(type(img),img.dtype,img.shape)
img_tinted = img *[1,0.95,0.9]
img_tinted = imresize(img_tinted,(300,300))
print(img_tinted.shape)
imsave('cat_tinted.jpg',img_tinted)
'''

'''
import numpy as np
from scipy.spatial.distance import pdist,squareform
#squareform化为对称矩阵格式
x = np.array([[0,1],[1,0],[2,0]])
print(x)
d = squareform(pdist(x))
s = squareform(np.array([1,2,3,4,5,6]))
print(s)
d2 = squareform(pdist(x.T))
print(d2)
'''

'''
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,3*np.pi,0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
plt.plot(x,y_sin)
plt.plot(x,y_cos)
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin and cos')
plt.legend(['sin','cos']) #要加[]!
plt.show()
plt.show()
'''

'''
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,3*np.pi,0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2,1,1) #几行，几列，占用多少空间
plt.plot(x,y_sin)
plt.title('sin')

plt.subplot(2,1,2)
plt.plot(x,y_cos)
plt.title('cos')

plt.show()
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread,imresize

img = imread('cat.jpeg')
img_tinted = img*[1,0.95,0.9]

plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(np.uint8(img_tinted)) #对np.array中的数据类型进行变换，需要类似于np.uint8()的操作
plt.show()
'''

'''
print(5/2)
print(5//2)
x = range(3)
print(type(x))
print(x)
for i in x:
    print(i)
print(list(x))
%%time
a = [i for i in range(100000)]
b = [i for i in range(100000)]
'''

'''
import numpy as np
nums = np.arange(8)
print(nums.shape)
print(nums.reshape((4,-1))) #自动匹配-1的那一行
nums = np.arange(8)
print(nums.min())
print(np.min(nums))
'''

'''
import numpy as np
x = np.array([[1,2],[3,4]],dtype = np.float64)
y = np.array([[5,6],[7,8]],dtype = np.float64)
print(x+y)
print(np.add(x,y))
print(x-y)
print(np.subtract(x,y))
print(x*y)
print(np.multiply(x,y))
print(np.sqrt(x))
print(x/y)
print(np.divide(x,y))
'''

'''
import numpy as np
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v,w))

print(x.dot(v))
print(np.dot(x,v))

print(x.dot(y))
print(np.dot(x,y))
'''

'''
import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print(np.sum(x))
print(np.sum(x,axis = 0))
print(np.sum(x,axis = 1))
print(np.max(x,axis = 1))
print(np.argmin(x,axis = 0))
print(x.max(axis = 0).shape)
x = np.array([[[1, 2, 3], 
               [4, 5, 6]],
              [[10, 23, 33], 
               [43, 52, 16]]
             ])
print(x.shape)
print(x.max(axis = 1).shape)
print(x.max(axis =(1,2))) #支持多维合并
'''

'''
import numpy as np
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print('original:\n',a)
print('element(0,0) a[0][0])',a[0][0])
print('element(0,0) a[0,0])',a[0,0])
b = a[:2,1:3]#修改b将会改变a，注意！
print('Sliced a[:2,1:3]\n',b)
print(a[0,:])
'''

'''
import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])
print(a)
b = np.array([0,2,0,1])
print(a[np.arange(4),b])
a[np.arange(4),b] += 10
print(a)

MAX = 5
nums = np.array([1,4,10,-1,15,0,5])
print(nums > MAX) #返回true/false np.array
print(nums[nums > MAX])
nums[nums > MAX] = 5
print(nums)

print(nums[[1,2,3,1,0]])
'''

'''
import numpy as np
x = np.array([[1,2,3],[3,5,7]])
print(x.shape)
col_means = x.mean(axis = 0)
print(col_means)
print(col_means.shape)
mean_shifted = x - col_means
print(mean_shifted)
print(mean_shifted.shape)
'''

'''
import numpy as np
x = np.array([[1,2,3],[3,5,7]])
print(x*2)
print(x.shape)
row_means = x.mean(axis = 1)
print(row_means)
print(row_means.shape)
mean_shifted = x -row_means.reshape((2,1))
print(mean_shifted)
print(mean_shifted.shape)
'''
'''
import numpy as np
v = np.array([1,2,3])
w = np.array([4,5])
print(v.reshape(3,1)*w)

x = np.array([[1,2,3],[4,5,6]])
print(x+v)
print((x.T+w).T)
print(x+np.reshape(w,(2,1)))
'''

'''
import numpy as np
x = np.arange(5)
view = x[1:3]
view[1] = -1
print(x)
x[2] = 10
print(view)
copy = x[[1,2]]
copy[1] = -1
print(copy)
print(x)
'''

'''
import numpy as np
x = np.arange(5)
print(x)

copy = x[x>2]
print(copy)
x[3] = 10
copy[0] = -1
print(x)
print(copy)
'''