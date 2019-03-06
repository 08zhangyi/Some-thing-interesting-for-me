import re

text_string = '文本最重要的来源无疑是网络。我们要把网络中的文本获取形成一个文本数据库。利用一个爬虫抓取到网络中的信息。爬取的策略有广度爬取和深度爬取。根据用户的需求，爬虫可以有主题爬虫和通用爬虫之分。'
regex = '^文本'
p_string = text_string.split('。')
for line in p_string:
    if re.search(regex, line) is not None:
        print(line)