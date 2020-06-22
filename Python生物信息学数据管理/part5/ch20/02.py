from Bio import ExPASy
from Bio import SeqIO

handle = ExPASy.get_sprot_raw("P04637")
seq_record = SeqIO.read(handle, "swiss")
print(seq_record.id)
print(seq_record.description)
out = open('myfile.fasta','w')
fasta = SeqIO.write(seq_record, out, "fasta")
out.close()