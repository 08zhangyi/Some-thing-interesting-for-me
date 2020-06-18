import re

seq = 'VSVLTMFRYAGWLDRLYMLVGTQLAAIIHGVALPLMMLI'
pattern = re.compile('[ST]Q')
match = pattern.search(seq)
if match:
    print('%10s' %(seq[match.start() - 4:match.end() + 4]))
    print('%6s' % match.group())
else:
    print("no match")

seq = 'QSAMGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQRPSKP'
pattern1 = re.compile('R(.)[ST][^P]')
match1 = pattern1.search(seq)
print(match1.group())
print(match1.group(1))
pattern2 = re.compile('R(.{0,3})[ST][^P]')
match2 = pattern2.search(seq)
print(match2.group())
print(match2.group(1))

separator = re.compile('\|')
annotation = 'ATOM:CA|RES:ALA|CHAIN:B|NUMRES:166'
columns = separator.split(annotation)
print(columns)
new_annotation = separator.sub('@', annotation)
print(new_annotation)
new_annotation2 = separator.sub('@', annotation, 2)
print(new_annotation2)
