from Bio import PDB

pdbl = PDB.PDBList()
pdbl.retrieve_pdb_file("2DN1")
parser = PDB.MMCIFParser()
structure = parser.get_structure("2DN1", "dn\\2dn1.cif")

for model in structure:
    for chain in model:
        print(chain)
        for residue in chain:
            print(residue.resname, residue.id[1])
            for atom in residue:
                print(atom.name, atom.coord)