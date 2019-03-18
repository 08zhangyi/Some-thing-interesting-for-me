'''
    论文A Structured Self-Attentive Sentence Embedding中网络结构的实现
'''
import torch as t
import numpy as np


class StructuredSelfAttentiveSentenceEmbedding(t.nn.Module):
    def __init__(self, word_numbers, word_embedding_size, hidden_size, num_layers, s1_size, s2_size):
        super().__init__()
        # 层定义
        self.embed_layer = t.nn.Embedding(word_numbers, word_embedding_size)
        self.lstm_layer = t.nn.LSTM(input_size=word_embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.ws1_layer = t.nn.Linear(2 * hidden_size, s1_size)
        self.ws1_tanh = t.nn.Tanh()
        self.ws2_layer = t.nn.Linear(s1_size, s2_size)
        self.ws2_softmax = t.nn.Softmax(1)

    def forward(self, word_index_tensor):
        # word_index_tensor为LongTensor类型，与embedding层的输入类似
        # word_index_tensor的size为batch_size * sentence_long
        word_embed = self.embed_layer(word_index_tensor)
        lstm_output, _ = self.lstm_layer(word_embed)  # 对双向的h对齐的输出
        # 计算attention权重
        a = self.ws1_layer(lstm_output)
        a = self.ws1_tanh(a)
        a = self.ws2_layer(a)
        a = self.ws2_softmax(a)
        # 根据attention权重合并计算s2_size个hdden_size的隐含向量和，为需要的特征向量
        a = t.unsqueeze(a, dim=3)
        lstm_output = t.unsqueeze(lstm_output, dim=2)
        m = a * lstm_output
        m = t.sum(m, dim=1)  # batch_size * s2_size * (2 * hidden_size)
        # 计算惩罚项
        a_t = a.transpose(2, 3)
        p = a * a_t
        p = t.sum(p, dim=1)
        im = t.Tensor(np.eye(p.size()[1])).unsqueeze(0)  # 单位矩阵的计算
        p = p - im
        p = t.pow(p, 2)
        p = t.sum(t.sum(p, dim=1), dim=1)  # batch_size
        return m, p, a


if __name__ == '__main__':
    sentences_index = [[0, 1, 2, 3], [0, 2, 3, 1], [3, 1, 2, 2]]
    sentences_index = t.LongTensor(sentences_index)
    # 模型参数设置
    word_embedding_size = 5
    hidden_size = 6
    num_layers = 1
    s1_size = 11
    s2_size = 7
    # 构建模型计算
    model = StructuredSelfAttentiveSentenceEmbedding(4, word_embedding_size, hidden_size, num_layers,  s1_size, s2_size)
    M, P, A = model(sentences_index)
    print(P)