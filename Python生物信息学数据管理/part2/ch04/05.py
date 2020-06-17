fasta_file = open("SwissProt.fasta", "r")
out_file = open("SwissPortHuman.fasta", "w")
seq = ''
for line in fasta_file:
    if line[0] == '>' and seq == '':
        header = line
    elif line[0] != '>':
        seq = seq + line
    elif line[0] == '>' and seq != '':
        if "Homo sapiens" in header:
            out_file.write(header + seq)
        seq = ''
        header = line
if "Homo sapiens" in header:
    out_file.write(header + seq)
out_file.close()
fasta_file.close()