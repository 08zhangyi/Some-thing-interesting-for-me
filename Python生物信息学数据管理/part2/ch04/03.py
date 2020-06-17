fasta_file = open('SwissProt.fasta', 'r')
out_file = open('SwissProt.header', 'w')
for line in fasta_file:
    if line[0] == '>':
        out_file.write(line)
fasta_file.close()
out_file.close()

input_file = open("SwissProt.fasta", "r")
ac_list = []
for line in input_file:
    if line[0] == '>':
        fields = line.split('|')
        ac_list.append(fields[1])
print(ac_list)