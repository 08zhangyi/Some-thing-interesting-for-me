from Bio import Entrez
from Bio import Medline

keyword = "PyCogent"
Entrez.email = "my_email@address.com"
handle = Entrez.esearch(db="pubmed", term=keyword)
record = Entrez.read(handle)
pmids = record['IdList']
print(pmids)
handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text")
medline_records = Medline.parse(handle)
records = list(medline_records)
n = 1
for record in records:
    if keyword in record["TI"]:
        print(n, ')', record["TI"])
        n += 1

handle = Entrez.einfo()
info = Entrez.read(handle)
print(info)
handle = Entrez.einfo(db="pubmed")
record = Entrez.read(handle)
print(record.keys())
print(record['DbInfo']['Description'])
print(record['DbInfo'])

handle = Entrez.esearch(db="pubmed", term="PyCogent AND RNA")
record = Entrez.read(handle)
print(record['IdList'])
handle = Entrez.esearch(db="pubmed", term="PyCogent OR RNA")
record = Entrez.read(handle)
print(record['Count'])
handle = Entrez.esearch(db="pubmed", term="PyCogent AND 2008[Year]")
record = Entrez.read(handle)
print(record['IdList'])
handle = Entrez.esearch(db="pubmed", term="C. elegans[Organism] AND 2008[Year] AND Mapk[Gene]")
record = Entrez.read(handle)
print(record['Count'])

handle = Entrez.esearch(db="nucleotide", term="Homo sapiens AND mRNA AND MapK")
records = Entrez.read(handle)
print(records['Count'])
top3_records = records['IdList'][0:3]
print(top3_records)
gi_list = ','.join(top3_records)
print(gi_list)
handle = Entrez.efetch(db="nucleotide", id=gi_list, rettype="gb", retmode="xml")
records = Entrez.read(handle)
print(len(records))
print(records[0].keys())
print(records[0]['GBSeq_organism'])

handle = Entrez.esearch(db="protein", term="Human AND cancer AND p21")
records = Entrez.read(handle)
print(records['Count'])
id_list = records['IdList'][0:3]
id_list = ",".join(id_list)
print(id_list)
handle = Entrez.efetch(db="protein", id=id_list, rettype="fasta", retmode="xml")
records = Entrez.read(handle)
rec = list(records)
print(rec[0].keys())
print(rec[0]['TSeq_defline'])
