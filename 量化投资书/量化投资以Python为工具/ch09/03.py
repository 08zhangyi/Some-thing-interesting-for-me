L1 = list([123, 23, [4, 5, 6], 'abc'])
print(L1)
L2 = list()
print(L2)
L2.append('python')
print(L2)
L2.extend([123, 'price', 7, 8, 9])
print(L2)
L2.append([123, 'price', 7, 8, 9])
print(L2)
L1 + ['python', 'price', 78]
print(L1)
L1 += ['python', 'price', 78]
print(L1)
L1.insert(4, 88)
print(L1)
L1.pop()
print(L1)
L1.remove('python')
print(L1)
# L1.remove('PYTHON')
L1.count('PYTHON')
L1.pop(3)
print(L1)

tu1 = tuple((123, 45, 6, 7, 8, 'python'))
print(type(tu1))
print(tu1)
tu2 = 123, 6, 7, 8, 'python'
print(type(tu2))
tu3 = (3, 4, 5, "python")
print(type(tu3))
tu4 = (2,)
print(type(tu4))
a = [34, 78.9, True, "Python", "finance"]
print(type(a))
b = tuple(a)
print(b)
print(type(b))
c = tuple("python")
print(c)
tu5 = (tu1, tu2, tu3, tu4)
print(tu5)
print(type(tu5))
a[0:2:4]
b[2:4]
len(c)
len(tu5)
tu5[0][3]
tu5[2]
tu5[2][3]
list2 = []
for i in tu5:
    list2.append(i)
    print('tuple:', i)
print(list2)
tu6 = (3**2, 4*5, 56/4, False)
print(tu6)
max(tu6)
min(tu6)
tu7 = tu4 + tu6
print(tu7)
print(tu4)
print(tu6)
tu6 * 3
print(tu6)

r1 = range(5)
type(r1)
len(r1)
print(r1)
r1[0:3]
list1 = list(r1)
print(list1)
tuple1 = tuple(range(2, 16, 3))
print(tuple1)
list2 = list(range(2, -9, -2))
print(list2)
# r1 * 2

st1 = 'Finance in python'
print(st1)
st2 = "Stock price of Alibab Group Holding Ltd"
print(st2)
st3 = """'beauty' of Python"""
print(st3)
st4 = str(123456)
print(st4)
len(st3)
st1[1:6]
st1[1:6:2]
st5 = 'py' + 'thon'
print(st5)
st4[0:2] + ' ' + 'price' + ' ' + st4[2:]
GeogeSoros = """I' m only rich because I know when I' m wrong, I basically have survived by recognizing my mistakes."""
GeogeSoros.split()
print(GeogeSoros)
'234,456,345'.split(',')
'234,456,345'.split(',', maxsplit=1)
'234,456,345'.split('3')
GeogeSoros.split(',', 2)
GeogeSoros = """I' m only rich because I know when I' m wrong, I basically have survived by recognizing my mistakes."""
split_GS = GeogeSoros.split()
count_GS = {}
for i in split_GS:
    if ',' in i:
        split_GS.remove(i)
        split_GS.append(i.split(',')[0])
    if '.' in i:
        split_GS.remove(i)
        split_GS.append(i.split('.')[0])
for j in split_GS:
    count_GS[j] = split_GS.count(j)
print(count_GS)
''.join(['a', 'b', 'c', 'd'])
','.join(['a', 'b', 'c', 'd'])
'A'.join(['a', 'b', 'c', 'd'])
'Finance'.islower()
'Finance'.lower()
'Stock price analysis'.title()
'Stockprice'.upper()
'Stockprice'.upper().title()
'the price is high.'.capitalize()
st2 = "stock price of Alibab Group Holding Ltd"
st2.capitalize()
st2[0:4].capitalize() + st2[4:]
'stock stocK Stock Stock,stock'.count('stock')
'stock stocK Stock Stock,stock'.lower().count('stock')