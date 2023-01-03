import time

startTime = time.time()
for trail in range(10000):
    building = ''
    for i in range(10000):
        building += 'x'
print('String concatenation: ', (time.time() - startTime))

startTime = time.time()
for trail in range(10000):
    building = []
    for i in range(10000):
        building.append('x')
building = ''.join(building)
print('List appending:       ', (time.time() - startTime))