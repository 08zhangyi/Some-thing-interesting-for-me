from Bio import PDB

parser = PDB.MMCIFParser()
structure = parser.get_structure("2DN1", "dn\\2dn1.cif")

model = structure[0]
chain = model['A']
residue_1 = chain[2]
residue_2 = chain[3]
atom_1 = residue_1['CA']
atom_2 = residue_2['CA']

dist = atom_1 - atom_2
print(dist)