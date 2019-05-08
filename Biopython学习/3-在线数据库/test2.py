# PubMed和Medline应用
from Bio import Entrez
Entrez.email = "A.N.Other@example.com"
handle = Entrez.egquery(term="orchid")  # 基本的搜索信息
print(handle.read())
handle = Entrez.esearch(db="pubmed", term="orchid", retmax=463)  # 搜索的具体内容
record = Entrez.read(handle)
print(record)
# 具体解析Medline结果
from Bio import Medline
id_list = ["19304878", "18606172", "16403221", "16377612", "14871861", "14630660"]
handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
records = Medline.parse(handle)
for record in records:
    print(record)

# 搜索下载Entrez核酸记录
handle = Entrez.egquery(term="Cypripedioideae")
record = Entrez.read(handle)
print(record)
handle = Entrez.esearch(db="nucleotide", term="Cypripedioideae", retmax=814)
record = Entrez.read(handle)
print(record)
# 下载部分结果
idlist = ",".join(record["IdList"][:5])
handle = Entrez.efetch(db="nucleotide", id=idlist, retmode="xml")
record = Entrez.read(handle)
print(record)

# 搜索下载GenBank记录
handle = Entrez.egquery(term="Opuntia AND rpl16")
record = Entrez.read(handle)
print(record)
handle = Entrez.esearch(db="nuccore", term="Opuntia AND rpl16")
record = Entrez.read(handle)
gi_list = record["IdList"]
gi_str = ",".join(gi_list)
handle = Entrez.efetch(db="nuccore", id=gi_str, rettype="gb", retmode="text")
text = handle.read()
print(text)
from Bio import SeqIO
handle = Entrez.efetch(db="nuccore", id=gi_str, rettype="gb", retmode="text")
records = SeqIO.parse(handle, "gb")
for record in records:
    print("%s, length %i, with %i features" % (record.name, len(record), len(record.features)))

# 物种关系谱图
handle = Entrez.esearch(db="Taxonomy", term="Cypripedioideae")
record = Entrez.read(handle)
handle = Entrez.efetch(db="Taxonomy", id="158330", retmode="xml")
records = Entrez.read(handle)
print(record)