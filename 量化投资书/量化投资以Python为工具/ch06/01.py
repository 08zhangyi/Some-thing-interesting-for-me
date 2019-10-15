a = 3 + 4
a = [1, 2, 3]
id(a)
b = a
id(b)
b[1] = 25
id(a)
id(b)
a
b
b[1] = 2
a
b
c = [1, 2, 3]
id(c)
c is a
c == a
c[2] = 88
c
a
b

x = y = z = 25678
x
y
z
x1 = y1 = z1 = [2, 5, 6, 7, 8]
x1
(x1 is y1) and (y1 is z1) and (x1 is z1)

x2, y2, z2 = 2, 5, 6
x2
y2
z2
x3 = 2, 5, 6
x3

a = 6
a *= 3
a
b = 17
b -= 6
b
c = 21
c %= 3
c

a = 6
b = 4
if a>b:
    print('变量a的值大于变量b的值')
else:
    print('变量b的值大于变量a的值')
if a>5 and b>5:
    a += 6
else:
    b += 6
a
b
grade = 95
if grade >= 90:
    print("Excellent job")
elif all([grade>70, grade<90]):
    print("good job")
else:
    print("It's uncommon")
a = 6
b = 4
'变量a的值大于变量b的值' if a>b else '变量b的值大于变量a的值'
a = 4**3 if {} else '123'
a

for i in [2, 3, 5, 6, 7]:
    print(i)
a = list()
for i in 'python':
    a.append(i+'python')
    print(a)
a = list()
for i in 'python':
    a.append(i+'python')
print(a)
a = [i+'python' for i in 'python']
a
list1 = [2, 10, 34, 3, 10, 20, 10]
[i for i in range(len(list1)) if list1[i] == 10]

a = 0
while a<4:
    a = a + 1
    print(a + 26)
print(a)

x = ['a', 'b', 'c']
y = [2, 3]
z = []
for i in x:
    for j in y:
        z.append([i, j])
print(z)
[[i, j] for i in x for j in y]

st1 = ['a', 'b', 'python', 'c', 'd']
for i in st1:
    print(i)
    if i == 'python':
        break
for i in st1:
    print(i)
    if i == 'python':
        break
    print('hello')
for i in st1:
    print(i)
    if i == 'python':
        break
print('hello')
for i in st1:
    if i == 'python':
        continue
    print(i)
print('hello')
for i in st1:
    if i == 'python':
        continue
    print(i)
    print('hello')
for i in st1:
    print(i)
    if i == 'python':
        continue
print('hello')
for i in st1:
    if i == 'python':
        continue
        print(i)
print('hello')
for i in range(5):
    if i>3:
        pass
    print([i, i+1])