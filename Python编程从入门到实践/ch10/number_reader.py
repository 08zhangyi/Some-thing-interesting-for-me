import json

filename = 'number.json'
with open(filename) as f:
    numbers = json.load(f)
print(numbers)