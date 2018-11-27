import torch as t


class StructuredSelfAttentiveSentenceEmbedding(t.nn.Module):
    def __init__(self, word_numbers, word_embedding_size, hidden_size, num_layers):
        self.embed_layer = t.nn.Embedding(word_numbers, word_embedding_size)
        self.lstm_layer = t.nn.LSTM(input_size=word_embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, word_index_tensor):
        # word_index_tensor为LongTensor类型，与embedding层的输入类似
        # word_index_tensor的size为batch_size * sentence_long
        word_embed = self.embed_layer(word_index_tensor)
        lstm_output, _ = self.lstm_layer(word_embed)  # 对双向的h对齐的输出


if __name__ == '__main__':
    pass