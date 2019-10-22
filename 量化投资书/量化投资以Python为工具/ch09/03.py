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

SSEC1 = {'Date': '02-Mar-2015', 'Open': 3332.7, 'High': 3336.8, 'Low': 3298.7, 'Close': 3336.3}
SSEC2 = dict({'Date': '02-Mar-2015', 'Open': 3332.7, 'High': 3336.8, 'Low': 3298.7, 'Close': 3336.3})
SSEC3 = dict(Date='02-Mar-2015', Open=3332.7, High=3336.8, Low=3298.7, Close=3336.3)
SSEC4 = dict([('Date', '02-Mar-2015'), ('Open', 3332.7), ('High', 3336.8), ('Low', 3298.7), ('Close', 3336.3)])
SSEC5 = dict(zip(['Data', 'Open', 'High', 'Low', 'Close'], ['02-Mar-2015', 3332.7, 3336.8, 3298.7, 3336.3]))
SSEC1 == SSEC2 == SSEC3 == SSEC4 == SSEC5
SSEC1.items()
SSEC1.keys()
SSEC1.values()
SSEC_keys = SSEC1.keys()
type(SSEC_keys)
SSEC_keysList = list(SSEC_keys)
SSEC_keysList
SSEC_keysList[2]
for v in SSEC1.values():
    if type(v) == float:
        v -= 20
        print((v+20, v))
    else:
        print(v, type(v))
SSEC1['Open']
SSEC1.get('Open')
for key in SSEC1.keys():
    if type(SSEC1[key])==float:
        SSEC1[key] += 100000
    print(key, ':', SSEC1.get(key))
SSEC1.update(Open=2332.7, High=2336.8, Low=2298.7, Close=2336.3)
SSEC1
SSEC1['index'] = '000001.SS'
del SSEC1['Date']
'Date' in SSEC1
SSEC1
SSEC2 = SSEC1.copy()
SSEC2.clear()
SSEC1
SSEC2

set1 = set([20, 50, 60, 34, 'python'])
print(set1)
list1 = [2, 3, 4, 5, 3, 2, 67]
set2 = set(list1)
print(set2)
fset1 = frozenset([23, 56, 'python'])
print(fset1)
set3 = {'Open', 'Close', 'High', 'Low'}
print(type(set3))
set1.add('finance')
print(set1)
set1.remove(20)
print(set1)
# fset1.add(3)
len(set3)
len(fset1)
list2 = [23.1, 24, 24.3, 22.9]
d = dict()
j = 0
for i in set3:
    d[i] = list2[j]
    j += 1
print(d)
set1 = {20, 30, 5, 6, 7}
set2 = {2, 3, 5, 7, 8}
set1.union(set2), set1 | set2
set1.intersection(set2), set1 & set2
set1.difference(set2), set1 - set2
set1.symmetric_difference(set2), set1 ^ set2
{1, 2, 3} <= {1, 2, 3}
{1, 2, 3} <= {1, 2, 3, 'python'}
{1, 2, 3, 'python'}.issuperset({1, 2, 3})
{5, 6, 7}.isdisjoint({})
{5, 6, 7}.isdisjoint({5})
{2, 3, 4, 7, 8, 9}.intersection({2, 3, 4, 6}).intersection({3, 4, 9})
{2, 3, 4, 7, 8, 9} & {2, 3, 4, 6} & {3, 4, 9}
{2, 3, 4, 7, 8, 9} - {2, 8} | {708, 100, 245}