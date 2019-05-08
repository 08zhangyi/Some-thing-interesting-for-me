from Bio.Blast import NCBIXML

# BLAST结果文件的解析
result_handle = open("data\\temp.xml")
# read适用于单条，parse适用于多条
blast_record = NCBIXML.read(result_handle)
print(blast_record)

# 打印结果
E_VALUE_THRESH = 0.05
for alignment in blast_record.alignments:
    for hsp in alignment.hsps:
        if hsp.expect < E_VALUE_THRESH:
            print('****Alignment****')
            print('sequence:', alignment.title)
            print('length:', alignment.length)
            print('e value:', hsp.expect)
            print(hsp.query[0:75] + '...')
            print(hsp.match[0:75] + '...')  # 匹配为竖杠，否则为空
            print(hsp.sbjct[0:75] + '...')
