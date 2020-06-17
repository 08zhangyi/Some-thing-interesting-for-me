InputFile = open("AY810830.gb")
OutputFile = open("AY810830.fasta", "w")
flag = 0
for line in InputFile:
    if line[0:9] == 'ACCESSION':
        AC = line.split()[1].strip()
        OutputFile.write('>' + AC + '\n')
    elif line[0:6] == 'ORIGIN':
        flag = 1
    elif flag == 1:
        fields = line.split()
        if fields != []:
            seq = ''.join(fields[1:])
            OutputFile.write(seq.upper() + '\n')
InputFile.close()
OutputFile.close()