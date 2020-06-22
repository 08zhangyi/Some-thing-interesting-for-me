from Bio import PDB
from Bio.PDB.Polypeptide import PPBuilder

parser = PDB.MMCIFParser()
structure = parser.get_structure("2DN1", "dn\\2dn1.cif")
ppb = PPBuilder()
peptides = ppb.build_peptides(structure)
for pep in peptides:
    print(pep.get_sequence())