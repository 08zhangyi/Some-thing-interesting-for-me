print(dir(__builtins__))
True
False
None
# english

import math
import cmath
print(math.pi)
math.sin(math.pi/2)
math.ceil(3.5)
math.floor(3.5)
math.trunc(3.5)
cmath.polar(1)
cmath.phase(complex(-1.0, 0.0))

import calendar
print(calendar.month(2016, 2))
calendar.isleap(2016)

import time
print(time.time())
print(time.localtime())
time.asctime()
time.ctime()
t1 = time.localtime(2000)
print(t1)
time.ctime(2000)
time.asctime(t1)

from time import strftime, strptime
t2 = strptime("12/02/2016", "%d/%m/%Y")
print(t2)
print(strftime("%d %b %y", t2))

from datetime import datetime
now = datetime.now()
print(now)
delta = datetime(2016, 2, 1) - datetime(2016, 1, 15)
print(delta)
now + delta
str(now)
now.strftime('%Y-%m-%d')
datetime.strptime('2016-01-01', '%Y-%m-%d')

from dateutil.parser import parse
parse('01-01-2016', dayfirst=True)
parse('01-01-2016')
parse('10')