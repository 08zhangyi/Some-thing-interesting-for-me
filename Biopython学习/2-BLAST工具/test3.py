from Bio import SearchIO

blast_qresult = SearchIO.read('data\\my_blast.xml', 'blast-xml')
print(blast_qresult)

for hit in blast_qresult:
    print(hit)  # 显示每段配对的基因库序列
# blast_qresult[3]  # 切片提取hit
# blast_qresult['gi|262205317|ref|NR_030195.1|']  # ID提取hit

# 排序
sort_key = lambda hit: hit.seq_len  # 排序比较方法
sorted_qresult = blast_qresult.sort(key=sort_key, reverse=True, in_place=False)

# 筛选
filter_func = lambda hit: len(hit.hsps) > 1   # 筛选条件
filtered_qresult = blast_qresult.hit_filter(filter_func)

# HSP(high-scoring pair)与Hit类似，Hit由一个或多个HSP组成
blast_hsp = blast_qresult[0][0]  # 提取一个HSP
print(blast_hsp)

# HSP片段
blast_frag = blast_qresult[0][0][0]
print(blast_frag)