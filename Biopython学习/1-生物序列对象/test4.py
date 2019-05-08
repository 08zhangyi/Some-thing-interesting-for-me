from Bio import AlignIO

# 单一序列对比
alignment = AlignIO.read("data\\PF05371_seed.sth", "stockholm")  # stockholm格式，还可以有PFAM格式
print(alignment)
for record in alignment:
    print("%s - %s" % (record.seq, record.id))  # SeqRecord格式
# alignment = AlignIO.read("data\\PF05371_seed.faa", "fasta")  # fasta格式比较文件
# alignments = AlignIO.parse("data\\resampled.phy", "phylip")  # 用来读取多个序列的对比

# 序列对比的输出
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
align1 = MultipleSeqAlignment([
             SeqRecord(Seq("ACTGCTAGCTAG", generic_dna), id="Alpha"),
             SeqRecord(Seq("ACT-CTAGCTAG", generic_dna), id="Beta"),
             SeqRecord(Seq("ACTGCTAGDTAG", generic_dna), id="Gamma"),
         ])
align2 = MultipleSeqAlignment([
             SeqRecord(Seq("GTCAGC-AG", generic_dna), id="Delta"),
             SeqRecord(Seq("GACAGCTAG", generic_dna), id="Epsilon"),
             SeqRecord(Seq("GTCAGCTAG", generic_dna), id="Zeta"),
         ])
align3 = MultipleSeqAlignment([
             SeqRecord(Seq("ACTAGTACAGCTG", generic_dna), id="Eta"),
             SeqRecord(Seq("ACTAGTACAGCT-", generic_dna), id="Theta"),
             SeqRecord(Seq("-CTACTACAGGTG", generic_dna), id="Iota"),
         ])
my_alignments = [align1, align2, align3]
AlignIO.write(my_alignments, "data\\my_example.phy", "phylip")

# 格式转换
# count = AlignIO.convert("data\\F05371_seed.sth", "stockholm", "data\\PF05371_seed.aln", "clustal")

# ClustalW多序列比较工具，需要单独安装，详见文档分析，旧
from Bio.Align.Applications import ClustalwCommandline, ClustalOmegaCommandline
# cline = ClustalwCommandline("clustalw2", infile="data\\opuntia.fasta")
cline = ClustalOmegaCommandline('D:\\programs\\自己玩的东西\\Biopython学习\\1-生物序列对象\\data\\clustal-omega-1.2.2-win64\\clustalo.exe',
                                infile="data\\opuntia.fasta",
                                outfile='data\\test1.aln',
                                guidetree_out='data\\test2.dnd',
                                force=True)
cline()
print(cline)

# MUSCLE序列比较工具，需要单独安装，详见文档分析，新
# 安装地址http://www.drive5.com/muscle/downloads.htm
from Bio.Align.Applications import MuscleCommandline
cline = MuscleCommandline('D:\\programs\\自己玩的东西\\Biopython学习\\1-生物序列对象\\data\\muscle.exe', input="opuntia.fasta", out="opuntia.txt")
print(cline)
cline = MuscleCommandline('D:\\programs\\自己玩的东西\\Biopython学习\\1-生物序列对象\\data\\muscle.exe', input="opuntia.fasta", out="opuntia.aln", clwstrict=True)
print(cline)