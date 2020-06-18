import struct


def convert_to_string(*args):
    '''returns all arguments as a single tab-separated string'''
    result = [str(a) for a in args]
    return '\t'.join(result) + '\n'


output_file = open("nucleotideSubstitMatrix.txt", "w")
output_file.write(convert_to_string('', 'A', 'T', 'C', 'G'))
output_file.write(convert_to_string('A', 1.0))
output_file.write(convert_to_string('T', 0.5, 1.0))
output_file.write(convert_to_string('C', 0.1, 0.1, 1.0))
output_file.write(convert_to_string('G', 0.1, 0.1, 0.5, 1.0))
output_file.close()

pdb_format = '6s5s1s4s1s3s1s1s4s1s3s8s8s8s6s6s10s2s3s'


def parse_atom_line(line):
    tmp = struct.unpack(pdb_format, line)
    atom = tmp[3].strip()
    res_type = tmp[5].strip()
    res_num = tmp[8].strip()
    chain = tmp[7].strip()
    x = float(tmp[11].strip())
    y = float(tmp[12].strip())
    z = float(tmp[13].strip())
    return chain, res_type, res_num, atom, x, y, z


def main(pdb_filename, residues, output_filename):
    pdb = open(pdb_filename)
    outfile = open(output_filename, "w")
    for line in pdb:
        if line.startswith("ATOM"):
            chain, res_type, res_num, atom, x, y, z = parse_atom_line(line)
            for aa, num in residues:
                if res_type == aa and res_num == num:
                    outfile.write(line)
    outfile.close()

residues = [('ASP', '102'), ('HIS', '57'), ('SER', '195')]
main("1TLD.pdb", residues, "trypsin_triad.pdb")