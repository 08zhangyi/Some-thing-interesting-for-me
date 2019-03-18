import torch as t

# 基本RNN的定义
input_size = 5
hidden_size = 10
num_layers = 2
rnn_layer = t.nn.RNN(input_size, hidden_size, num_layers, bidirectional=False)
# 输入张量的定义
seq_len = 6
batch_size = 3
input_size = 5
x = t.randn(seq_len, batch_size, input_size)
# h0的定义，默认为0向量
h0 = t.zeros(num_layers * 1, batch_size, hidden_size)  # 单向时num_layers * 1，双向时num_layers * 2
output, hn = rnn_layer(x)  # output为所有时刻的输出，size为seq_len * batch_size * hidden_size，hn为最后时刻的hidden单元，size与h0相同
# output[-1, :, :]与hn相同

# LSTM的定义
lstm_layer = t.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
# h0、c0的定义，默认为0向量
h0 = t.zeros(num_layers * 2, batch_size, hidden_size)
c0 = t.zeros(num_layers * 2, batch_size, hidden_size)
output, (hn, cn) = lstm_layer(x)
print(output.size())

# 控制RNN Cell的使用，手动写RNN的实现
rnn_cell = t.nn.RNNCell(input_size, hidden_size)
h = t.zeros(batch_size, hidden_size)  # 隐含层h的初始化
output = []
for i in range(seq_len):
    h = rnn_cell(x[i], h)
    output.append(h)
print(output)