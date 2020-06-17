propensities = {
   'N': 0.2299, 'P': 0.5523, 'Q':-0.18770, 'A':-0.2615,
   'R':-0.1766, 'S': 0.1429, 'C':-0.01515, 'T': 0.0089,
   'D': 0.2276, 'E':-0.2047, 'V':-0.38620, 'F':-0.2256,
   'W':-0.2434, 'G': 0.4332, 'H':-0.00120, 'Y':-0.2075,
   'I':-0.4222, 'K':-0.1001, 'L': 0.33793, 'M':-0.2259
   }
threshold = 0.3
input_seq = "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSG\
IQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSR\
VASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYP\
GQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYT\
KVCNYVSWIKQTIASN"
output_seq = ""
for res in input_seq:
   if res in propensities:
      if propensities[res] >= threshold:
         output_seq += res.upper()
      else:
         output_seq += res.lower()
   else:
      print('unrecognized character:', res)
      break
print(output_seq)

aa_codes = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}
seq = ''
for line in open("1TLD.pdb"):
    if line[0:6] == "SEQRES":
        columns = line.split()
        for resname in columns[4:]:
            seq = seq + aa_codes[resname]
i = 0
print(">1TLD")
while i < len(seq):
    print(seq[i:i + 64])
    i = i + 64

