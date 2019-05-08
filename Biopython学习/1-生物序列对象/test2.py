from Bio.SeqRecord import SeqRecord

# SeqwRecord的属性：
# .seq
# .id，序列主ID
# .name
# .description
# .letter_annotations，对照序列的每个字母的解释，以信息名为键，信息内容为值，值与序列长度一致
# .annotations，附加信息字典
# .features，SeqFeature对象列表，存储结构化信息
# .dbxrefs，数据库交叉引用信息列表

# 新建SeqRecord
from Bio.Seq import Seq
# 最简单的例子
simple_seq = Seq("GATC")
simple_seq_r = SeqRecord(simple_seq)
simple_seq_r.id = "AC12345"  # 改动id
simple_seq_r.description = "Made up sequence I wish I could write a paper about"
simple_seq_r.annotations["evidence"] = "None. I just made it up."
simple_seq_r.letter_annotations["phred_quality"] = [40,40,38,30]
print(simple_seq_r)
# 根据FASTA文件创建
from Bio import SeqIO
record = SeqIO.read("data\\NC_005816.fna", "fasta")  # 读取文件
print(record)
print(record.id)
print(record.name)
print(record.description)
# 根据GenBank文件创建
record = SeqIO.read("data\\NC_005816.gb", "genbank")
print(record)
print(record.id)
print(record.name)
print(record.description)
print(record.features)

# SeqFeature对象
# .type，用文字描述的feature类型 (如 ‘CDS’ 或 ‘gene’)
# .location，在序列中所处的位置
# .qualifiers，存储feature附加信息（Python字典）
# .sub_features

# 格式化方法
from Bio.Alphabet import generic_protein
record = SeqRecord(Seq("MMYQQGCFAGGTVLRLAKDLAENNRGARVLVVCSEITAVTFRGPSETHLDSMVGQALFGD" \
                      +"GAGAVIVGSDPDLSVERPLYELVWTGATLLPDSEGAIDGHLREVGLTFHLLKDVPGLISK" \
                      +"NIEKSLKEAFTPLGISDWNSTFWIAHPGGPAILDQVEAKLGLKEEKMRATREVLSEYGNM" \
                      +"SSAC", generic_protein),
                   id="gi|14150838|gb|AAK54648.1|AF376133_1",
                   description="chalcone synthase [Cucumis sativus]")
record.format("fasta")  # 转化为fasta格式