from Bio import PDB
from Bio.PDB import PDBIO

parser = PDB.MMCIFParser()
structure = parser.get_structure("2DN1", "dn\\2dn1.cif")

io = PDBIO()
io.set_structure(structure)
io.save('my_structure.pdb')
