from Bio.ExPASy import ScanProsite, Prosite, Prodoc
from Bio import ExPASy

# 在线浏览蛋白质序列
sequence = "MEHKEVVLLLLLFLKSGQGEPLDDYVNTQGASLFSVTKKQLGAGSIEECAAKCEEDEEFTCRAFQYHSKEQQCVIMAENRKSSIIIRMRDVVLFEKKVYLSECKTGNGKNYRGTMSKTKN"
handle = ScanProsite.scan(seq=sequence)
result = ScanProsite.read(handle)
print(result)

# 获取Prosite记录
handle = ExPASy.get_prosite_raw('PS00001')
record = Prosite.read(handle)
# record = Prodoc.read(handle)
print(record)