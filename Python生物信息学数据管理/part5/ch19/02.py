from Bio import Seq
from Bio.Alphabet import IUPAC

seq = Seq.Seq("AGCATCGTAGCATG", IUPAC.unambiguous_dna)
print(seq[5])

mutable = Seq.MutableSeq("AGCATCGTAGCATG", IUPAC.unambiguous_dna)
mutable[5] = "T"
print(mutable)
