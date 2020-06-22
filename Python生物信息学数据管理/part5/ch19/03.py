from Bio import SeqIO

fasta_file = open("Uniprot.fasta", "r")
for seq_record in SeqIO.parse(fasta_file, "fasta"):
    print(seq_record.id)
    print(repr(seq_record.seq))
    print(len(seq_record))
fasta_file.close()

genbank_file = open ("AY810830.gbk", "r")
output_file = open("AY810830.fasta", "w")
records = SeqIO.parse(genbank_file, "genbank")
SeqIO.write(records, output_file, "fasta")
output_file.close()