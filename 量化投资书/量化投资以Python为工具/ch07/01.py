def my_subtraction(minuend, subtrahend):
    difference = minuend - subtrahend
    return difference


i = 6; j = 2
d = my_subtraction(i, j)
print(d)
d2 = my_subtraction(7, 2.0)
print(d2)


def my_abs(x):
    if x>=0:
        return x
    else:
        return -x


a = 1
b = 2
my_abs(a - b)


def hello():
    print('Hello World!')


hello()


def donothing():
    pass


def my_arithmetic(x, y):
    z1 = x + y
    z2 = x - y
    return z1, z2


r1, r2 = my_arithmetic(7, 5)
print(r1, r2)

my_subtraction(7, 2)
my_subtraction(2, 7)
my_subtraction(minuend=7, subtrahend=2)
my_subtraction(subtrahend=2, minuend=7)


def my_print(arg1, arg2='World!'):
    print(arg1, arg2)


my_print('Hi')
expression = 'Hi'


def greeting(words=expression):
    print(words)


greeting()
expression = 'Hello'
greeting()


def change_obj(x, y):
    x[0] = 'A'
    y = 7


letters = ['a', 'b', 'c']
number = 6
change_obj(letters, number)
print(letters, number)


def growing_list(x, y=[]):
    y.append(x)
    print(y)


growing_list('a')
growing_list('b')
growing_list('c')
growing_list('d')


def my_addition0(addend):
    sum = 0
    for i in addend:
        sum = sum + i
    return sum


numbers0 = (1, 2, 3)
numbers1 = [1, 2, 3]
my_addition0(numbers0)
my_addition0(numbers1)


def my_addition1(*addend):
    sum = 0
    for i in addend:
        sum = sum + i
    return sum


my_addition1(1, 2)
my_addition1(1, 2, 3)


def weighted_sum(x1, x2, *y):
    sum = 0
    size = len(y)
    weight = 0.3 / size
    for i in y:
        sum = sum + weight*i
    sum = sum + 0.4*x1 + 0.3*x2
    return sum


weighted_sum(6, 7, 8, 9, 10)


def greeting2():
    print('Hello World!')


greeting2()
greeting3 = lambda : print('Hello World!')
greeting3()
def power2(x): return x**2
def power3(x): return x**3
def power4(x): return x**4
def power5(x): return x**5
L1 = [power2, power3, power4, power5]
for p in L1:
    print(p(3))

x = 6
x + 3


def fun1(value):
    return (x + value)


fun1(7)


def fun2():
    y = 10


# x + y


def fun3(value):
    x = 60
    return (x + value)


fun3(7)
a = 10


def fun4():
    global a
    a = 20


fun4()
print(a)
