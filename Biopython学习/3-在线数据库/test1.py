from Bio import Entrez

# 访问NCBI Entrez数据库
Entrez.email = "A.N.Other@example.com"  # 先设置一个全局邮件地址
Entrez.tool = "MyLocalScript"  # 配置环境必须的语句

# 数据库基本信息
handle = Entrez.einfo()
record = Entrez.read(handle)  # 解析XML格式
print(record)

# 搜索内容
# 搜索PubMed
handle = Entrez.esearch(db="pubmed", term="biopython")
record = Entrez.read(handle)
print(record)
# 搜索GenBank，Cypripedioideae orchids中的matK gene
handle = Entrez.esearch(db="nucleotide",term="Cypripedioideae[Orgn] AND matK[Gene]")
record = Entrez.read(handle)
print(record)

# 上传内容，反馈具体信息
id_list = ["19304878", "18606172", "16403221", "16377612", "14871861", "14630660"]
handle = Entrez.epost("pubmed", id=",".join(id_list))
record = Entrez.read(handle)
print(record)

# 通过ID获取摘要
handle = Entrez.esummary(db="PubMed", id="30367")
record = Entrez.read(handle)
print(record)

# 下载全面的信息
handle = Entrez.efetch(db="nucleotide", id="186972394", rettype="gb", retmode="text")
# rettype="fasta"返回fasta模式，retmode默认返回XML格式
# from Bio import SeqIO
# record = SeqIO.read(handle, "genbank")  # 返回结果直接读入SeqRecord中
print(handle.read())

# 搜索相关条目
handle = Entrez.elink(dbfrom="pubmed", id="19304878")
record = Entrez.read(handle)
print(record)

# 搜索相关条目统计信息汇总
handle = Entrez.egquery(term="biopython")
record = Entrez.read(handle)
print(record)

# 专用解析器，解析不同的专用格式
from Bio import Medline  # PubMed中的MEDLINE格式
from Bio import Geo  # 高通量基因表达和杂交芯片数据的数据库Gene Expression Omnibus
from Bio import UniGene  # UniGene是NCBI的转录组数据库，每个UniGene记录展示了该转录本在某个特定物种中相关的基因