pi = 3.141592653589793


def findMax(sequence):
    maxValue = sequence[0]
    for a in sequence:
        if a > maxValue:
            maxValue = a
    return (maxValue)


def findMin(sequence):
    minValue = sequence[0]
    for a in sequence:
        if a < minValue:
            minValue = a
    return (minValue)