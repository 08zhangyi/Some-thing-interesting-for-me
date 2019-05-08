from Bio.Seq import Seq

my_seq = Seq("AGTACACTGGT")
print(my_seq)
print(my_seq.alphabet)

from Bio.Alphabet import IUPAC
my_seq = Seq("AGTACACTGGT", IUPAC.unambiguous_dna)
print(my_seq)
print(my_seq.alphabet)

my_prot = Seq("AGTACACTGGT", IUPAC.protein)
print(my_prot)
print(my_prot.alphabet)

# 计数操作
print(my_seq.count('A'))

# 序列相加
from Bio.Alphabet import generic_nucleotide
nuc_seq = Seq("GATCGATGC", generic_nucleotide)
dna_seq = Seq("ACGT", IUPAC.unambiguous_dna)
print(nuc_seq + dna_seq)

# 核苷酸互补序列
my_seq = Seq("GATCGATGGGCCTATATAGGATCGAAAATCGC", IUPAC.unambiguous_dna)
print(my_seq)
print(my_seq.complement())
print(my_seq.reverse_complement())

# 转录
coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", IUPAC.unambiguous_dna)
messenger_rna = coding_dna.transcribe()  # 直接转换为RNA
messenger_rna.back_transcribe()  # 转换回DNA
print(messenger_rna)
template_dna = coding_dna.reverse_complement()  # 逆序互补
template_dna.reverse_complement().transcribe()  # 生物学意义上的转录，两步

# 翻译
messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG", IUPAC.unambiguous_rna)  # 从RNA开始
print(messenger_rna.translate())
coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG", IUPAC.unambiguous_dna)  # 从DNA开始
print(coding_dna.translate())
# 默认使用NCBI标准翻译表，可以用table参数给出不同的翻译表
print(coding_dna.translate(table="Vertebrate Mitochondrial"))  # 用线粒体
print(coding_dna.translate(table=2))  # 用表格编号
coding_dna.translate(to_stop=True)  # 到第一个终止码子翻译结束，终止密码子本身不翻译
coding_dna.translate(table=2, stop_symbol="@")  # 指定终止符号，默认为*
# 更特殊的使用
from Bio.Alphabet import generic_dna
gene = Seq("GTGAAAAAGATGCAATCTATCGTACTCGCACTTTCCCTGGTTCTGGTCGCTCCCATGGCA" +\
           "GCACAGGCTGCGGAAATTACGTTAGTCCCGTCAGTAAAATTACAGATAGGCGATCGTGAT" +\
           "AATCGTGGCTATTACTGGGATGGAGGTCACTGGCGCGACCACGGCTGGTGGAAACAACAT" +\
           "TATGAATGGCGAGGCAATCGCTGGCACCTACACGGACCGCCGCCACCGCCGCGCCACCAT" +\
           "AAGAAAGCTCCTCATGATCATCACGGCGGTCATGGTCCAGGCAAACATCACCGCTAA",
           generic_dna)
print(gene.translate(table="Bacterial", cds=True))  # 细菌密码中，GTG在正常和作为起始密码子时意义不一样，CDS=True告诉此序列是完整的CDS序列

# 翻译表
# 翻译表名称参考网址ftp://ftp.ncbi.nlm.nih.gov/entrez/misc/data/gc.prt或https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi
from Bio.Data import CodonTable
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]  # 引入标准翻译表
mito_table = CodonTable.unambiguous_dna_by_name["Vertebrate Mitochondrial"]  # 引入线粒体翻译表
# standard_table = CodonTable.unambiguous_dna_by_id[1]
# mito_table = CodonTable.unambiguous_dna_by_id[2]
print(standard_table)
print(mito_table)
print(mito_table.stop_codons)
print(mito_table.start_codons)

# 可修改的seq
my_seq = Seq("GCCATTGTAATGGGCCGCTGAAAGGGTGCCCGA", IUPAC.unambiguous_dna)
mutable_seq = my_seq.tomutable()
print(mutable_seq)  # MutableSeq对象
from Bio.Seq import MutableSeq
mutable_seq = MutableSeq("GCCATTGTAATGGGCCGCTGAAAGGGTGCCCGA", IUPAC.unambiguous_dna)
mutable_seq[5] = "C"
mutable_seq.reverse()
new_seq = mutable_seq.toseq()  # 回到只读对象