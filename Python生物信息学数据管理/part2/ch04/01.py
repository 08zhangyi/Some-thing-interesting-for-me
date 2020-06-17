list_a = []
for line in open("cell_cycle_proteins.txt"):
    list_a.append(line.strip())
print(list_a)
list_b = []
for line in open("cancer_cell_proteins.txt"):
    list_b.append(line.strip())
print(list_b)
for protein in list_a:
    if protein in list_b:
        print(protein, 'detected in the cancer cell')
    else:
        print(protein, 'not observed')