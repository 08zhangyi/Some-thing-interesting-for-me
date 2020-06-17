input_file = open('UniprotID.txt')
output_file = open('UniprotID-unique.txt', 'w')
unique = []
for line in input_file:
    if line not in unique:
        output_file.write(line)
        unique.append(line)

input_file.close()
output_file.close()

input_file = open('UniprotID.txt')
output_file = open('UniprotID-unique.txt','w')
unique = set(input_file)
for line in unique:
    output_file.write(line)