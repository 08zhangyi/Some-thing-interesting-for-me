from Bio.Blast import NCBIWWW

# 使用GI号搜索
print(1)
result_handle = NCBIWWW.qblast("blastn", "nt", "8332116")
print(result_handle.read())
# 也可以使用fasta文件
from Bio import SeqIO
print(2)
record = SeqIO.read('data\\NC_005816.gb', format="genbank")
# result_handle = NCBIWWW.qblast("blastn", "nt", record.seq)
result_handle = NCBIWWW.qblast("blastn", "nt", record.format("fasta"))  # 以fasta格式返回字符串
# 保存文件，read()只能用一次
save_file = open('data\\temp.xml', 'w')
save_file.write(result_handle.read())
save_file.close()
