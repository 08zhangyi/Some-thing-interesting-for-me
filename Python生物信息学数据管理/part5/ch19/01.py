from Bio import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

dna = open('hemoglobin-gene.txt').read().strip()
dna = Seq.Seq(dna, IUPAC.unambiguous_dna)
mrna = dna.transcribe()
protein = mrna.translate()
protein_record = SeqRecord(protein, id='sp|P69905.2|HBA_HUMAN', description="Hemoglobin subunit alpha, human")
outfile = open("HBA_HUMAN.fasta", "w")
SeqIO.write(protein_record, outfile, "fasta")
outfile.close()
