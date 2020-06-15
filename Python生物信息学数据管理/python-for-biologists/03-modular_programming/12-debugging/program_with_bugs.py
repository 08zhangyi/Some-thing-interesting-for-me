'''

A program with bugs. Try to fix them all!

-----------------------------------------------------------
(c) 2013 Allegra Via and Kristian Rother
    Licensed under the conditions of the Python License

    This code appears in section 12.2.2 of the book
    "Managing Biological Data with Python".
-----------------------------------------------------------
'''

def evaluate_data(data, lower=100, upper=300):
    """Counts data points in three bins."""
    smaller = 0
    between = 0
    bigger  = 0
    
    for length in data:
        if length < lower:
            smaller = smaller + 1
        elif lower < length < upper:
            between = between + 1
        elif length > upper:
            bigger += 1
    return smaller, between, bigger

def read_data(filename):
    """Reads neuron lengths from a text file."""
    primary, secondary = [], []

    for line in open(filename):
        category, length = line.split("\t")
        length = float(length)
        if category == "Primary":
            primary.append(length)
        elif category == "Secondary":
            secondary.append(length)
    return primary, secondary

def write_output(filename, count_pri, count_sec):
    """Writes counted values to a file."""
    output = open(filename,"w")
    output.write("category      <100  100-300   >300\n")
    output.write("Primary  :  %5i   %5i   %5i\n" % count_pri)
    output.write("Secondary:  %5i   %5i   %5i\n" % count_sec)
    output.close()

primary, secondary = read_data('neuron_data.txt')
print primary
count_pri = evaluate_data(primary)
print count_pri,"AAA"
count_sec = evaluate_data(secondary)
print count_sec,"BBB"
write_output('results.txt' , count_pri,count_sec)
