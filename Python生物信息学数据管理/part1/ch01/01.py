import math

ATP = 3.5
ADP = 1.8
Pi = 5.0
R = 0.00831
T = 298
deltaG0 = -30.5
print(deltaG0 + R * T * math.log(ADP * Pi / ATP))

diameter = 30.0
radius = diameter / 2.0
length = 115.0
print(math.pi * radius ** 2 * length / 1000.0)

x1, y1, z1 = 0.1, 0.0, -0.7
x2, y2, z2 = 0.5, -1.0, 2.7
dx = x1 - x2
dy = y1 - y2
dz = z1 - z2
dsquare = math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2)
d = math.sqrt(dsquare)
print(d)