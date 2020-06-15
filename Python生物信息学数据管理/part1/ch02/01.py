import random

insulin = "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
for amino_acid in "ACDEFGHIKLMNPQRSTVWY":
    number = insulin.count(amino_acid)
    print(amino_acid, number)

for character in 'hemoglobin':
    print(character, end='')

alphabet = 'AGCT'
sequence = ""
for i in range(10):
    index = random.randint(0, 3)
    sequence = sequence + alphabet[index]
print(sequence)

seq = "PRQTEINSEQWENCE"
for i in range(len(seq)-4):
    print(seq[i:i+5])