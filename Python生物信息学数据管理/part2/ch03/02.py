import math

data = [3.53, 3.47, 3.51, 3.72, 3.43]
average = sum(data) / len(data)
total = 0.0
for value in data:
    total += (value - average) ** 2
stddev = math.sqrt(total / len(data))
print(stddev)

data = [3.53, 3.47, 3.51, 3.72, 3.43]
data.sort()
mid = len(data) / 2
if len(data) % 2 == 0:
    median = (data[mid - 1] + data[mid]) / 2.0
else:
    median = data[mid]
print(median)