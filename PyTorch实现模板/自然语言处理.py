import torch as t

# 定义词词典，从word转化为index
# 一般从文本给出，此处假设已经给定
word_to_index = {'hello': 0, 'world': 1, 'liu':2, 'he': 3}

Word_number = len(word_to_index)  # 词汇数量
Embedding_dim = 5  # 嵌入维数
embeds = t.nn.Embedding(Word_number, Embedding_dim)  # 定义词嵌入层
idx_tensor = t.LongTensor([word_to_index['hello']])  # 用LongTensor输入
embed_tensor = embeds(idx_tensor)

# 定义batch为3的一个样本
sentences = [['hello', 'liu', 'world', 'he'], ['hello', 'liu', 'world', 'he'], ['he', 'liu', 'world', 'hello']]
sentences_index = [[word_to_index[w] for w in s] for s in sentences]
idx_tensor = t.LongTensor(sentences_index)
embed_layer = t.nn.Embedding(Word_number, Embedding_dim)
embed_tensor = embed_layer(idx_tensor).size()  # embed_tensor的size为batch_size * sentence_long * embedding_dim