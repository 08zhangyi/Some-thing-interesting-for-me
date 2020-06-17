swissprot = open("SwissProt.fasta")
insulin_ac = 'P61981'
result = None
while result == None:
    line = next(swissprot)
    if line.startswith('>'):
        ac = line.split('|')[1]
        if ac == insulin_ac:
            result = line.strip()
print(result)