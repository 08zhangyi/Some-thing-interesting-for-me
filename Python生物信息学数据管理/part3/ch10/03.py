from math import sqrt
import struct


def calc_dist(p1, p2):
    '''
    Returns the distance between two 3D points.
    '''
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    distsq = pow(dx, 2) + pow(dy, 2) + pow(dz, 2)
    distance = sqrt(distsq)
    return distance


pdb_format = '6s5s1s4s1s3s1s1s4s1s3s8s8s8s6s6s10s2s3s'


def parse_atom_line(line):
    '''Returns an ATOM line parsed to a tuple '''
    tmp = struct.unpack(pdb_format, line)
    atom = tmp[3].strip()
    res_type = tmp[5].strip()
    res_num = tmp[8].strip()
    chain = tmp[7].strip()
    x = float(tmp[11].strip())
    y = float(tmp[12].strip())
    z = float(tmp[13].strip())
    return chain, res_type, res_num, atom, x, y, z


pdb = open('3G5U.pdb')
points = []
while len(points) < 2:
    line = pdb.readline()
    if line.startswith("ATOM"):
        chain, res_type, res_num, atom, x, y, z = parse_atom_line(line)
        if res_num == '123' and chain == 'A' and atom == 'CA':
            points.append((x, y, z))
        if res_num == '209' and chain == 'A' and atom == 'CA':
            points.append((x, y, z))
print(calc_dist(points[0], points[1]))


def get_ca_atoms(pdb_filename):
    pdb_file = open(pdb_filename, "r")
    ca_list = []
    for line in pdb_file:
        if line.startswith('ATOM'):
            data = parse_atom_line(line)
            chain, res_type, res_num, atom, x, y, z = data
            if atom == 'CA' and chain == 'A':
                ca_list.append(data)
    pdb_file.close()
    return ca_list


ca_atoms = get_ca_atoms("1TLD.pdb")
for i, atom1 in enumerate(ca_atoms):
    name1 = atom1[1] + atom1[2]
    coord1 = atom1[4:]
    for j in range(i+1, len(ca_atoms)):
        atom2 = ca_atoms[j]
        name2 = atom2[1] + atom2[2]
        coord2 = atom2[4:]
        dist = calc_dist(coord1, coord2)
        print(name1, name2, dist)