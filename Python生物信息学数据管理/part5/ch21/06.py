from Bio import PDB
from Bio.PDB import PDBIO

parser = PDB.MMCIFParser()
structure = parser.get_structure("2DN1", "dn\\2dn1.cif")

atom1 = structure[0]["A"][10]["CA"]
atom2 = structure[0]["A"][20]["CA"]
atom3 = structure[0]["A"][30]["CA"]
atom4 = structure[0]["B"][10]["CA"]
atom5 = structure[0]["B"][20]["CA"]
atom6 = structure[0]["B"][30]["CA"]

moving = [atom1, atom2, atom3]
fixed = [atom4, atom5, atom6]

sup = PDB.Superimposer()
sup.set_atoms(fixed, moving)
print(sup.rotran)
print('RMS:', sup.rms)

chainA = structure[0]['A']
chainA.transform(sup.rotran[0], sup.rotran[1])

io = PDBIO()
io.set_structure(structure)
io.save('my_structure_temp.pdb')