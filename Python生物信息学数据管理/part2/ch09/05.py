import urllib.request
import re

word_regexp = re.compile('schistosoma')
pmids = ['18235848', '22607149', '22405002', '21630672']
for pmid in pmids:
    url = 'http://www.ncbi.nlm.nih.gov/pubmed?term=' + pmid
    handler = urllib.request.urlopen(url)
    html = handler.read()
    title_regexp = re.compile('<h1>.{5,400}</h1>')
    title = title_regexp.search(html)
    title = title.group()
    abstract_regexp = re.compile('<h3>Abstract</h3><p>.{20,3000}</p></div>')
    abstract = abstract_regexp.search(html)
    abstract = abstract.group()
    word = word_regexp.search(abstract, re.IGNORECASE)
    if word:
        print(title)
        print(word.group(), word.start(), word.end())