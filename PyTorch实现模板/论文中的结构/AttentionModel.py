# 文献上汇总的Attention模型的实现，用于自然语言处理

import torch as t


class AttentionLayerV1(t.nn.Module):
    def __init__(self, INPUT_DIM, Z_DIM):
        super().__init__()
        self.w = t.nn.Parameter(t.randn(INPUT_DIM, Z_DIM))

    def forward(self, h, z):
        # h为输入的序列，(SEQ_LEN, BATCH_SIZE, INPUT_DIM)
        # z为隐含需要提取的信息，(BATCH_SIZE, Z_DIM)
        h_a = t.matmul(h, self.w)  # w对输入h变形，得到h_a
        z = z.unsqueeze(0)
        # 得到h_a与z对应位置做內积，计算相似性
        h_a = h_a * z
        h_a = t.sum(h_a, 2)
        # 计算Attention权重
        alpha = t.nn.functional.softmax(h_a, 0).unsqueeze(2)
        # 权重求和计算注意力权重
        output = h * alpha
        output = t.sum(output, 0)  # (BATCH_SIZE, INPUT_DIM)
        return output


class AttentionLayerV2(t.nn.Module):
    def __init__(self, INPUT_DIM, Z_SIZE, Z_DIM):
        super().__init__()
        self.w = t.nn.Parameter(t.randn(1, 1, Z_SIZE, INPUT_DIM, Z_DIM))

    def forward(self, h, z):
        # h为输入的序列，(SEQ_LEN, BATCH_SIZE, INPUT_DIM)
        # z为隐含需要提取的信息，(BATCH_SIZE, Z_SIZE, Z_DIM)，Z_SIZE为提取信息的个数
        h_a = h.unsqueeze(2).unsqueeze(4)
        h_a = t.sum(h_a * self.w, 3)
        z = z.unsqueeze(0)
        # 得到h_a与z对应位置做內积，计算相似性
        h_a = h_a * z
        h_a = t.sum(h_a, 3)
        # 计算Attention权重
        alpha = t.nn.functional.softmax(h_a, 0).unsqueeze(3)
        # 权重求和计算注意力权重
        output = h.unsqueeze(2) * alpha
        output = t.sum(output, 0)  # (BATCH_SIZE, Z_SIZE, INPUT_DIM)
        return output


class AttentionLayerGoogle(t.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        # Q的size为n * batch_size * dk
        # K的size为m * batch_size * dk
        # V的size为m * batch_size * dv
        # n为sequence_length，m为Attention个数
        Q = Q.unsqueeze(1)
        K = K.unsqueeze(0)
        QK = t.sum(Q * K, 3) / K.size()[2]
        QK = t.nn.Softmax(1)(QK)  # QK = softmax(QK/sqrt(dk))
        QK = QK.unsqueeze(3)
        V = V.unsqueeze(0)
        output = t.sum(QK * V, 1)
        return output


class MultiHeadAttentionLayerGoogle(t.nn.Module):
    def __init__(self, head_num, dk_input, dv_input, dk_output, dv_output):
        super().__init__()
        self.WQ = t.nn.ParameterList([t.nn.Parameter(t.randn(1, 1, dk_input, dk_output)) for _ in range(head_num)])
        self.WK = t.nn.ParameterList([t.nn.Parameter(t.randn(1, 1, dk_input, dk_output)) for _ in range(head_num)])
        self.WV = t.nn.ParameterList([t.nn.Parameter(t.randn(1, 1, dv_input, dv_output)) for _ in range(head_num)])
        self.attention_layers = [AttentionLayerGoogle() for _ in range(head_num)]
        self.head_num = head_num

    def forward(self, Q, K, V):
        Q_list = [t.sum(Q.unsqueeze(3) * self.WQ[i], 2) for i in range(self.head_num)]
        K_list = [t.sum(K.unsqueeze(3) * self.WK[i], 2) for i in range(self.head_num)]
        V_list = [t.sum(V.unsqueeze(3) * self.WV[i], 2) for i in range(self.head_num)]
        output_list = [self.attention_layers[i](Q_list[i], K_list[i], V_list[i]) for i in range(self.head_num)]
        output = t.cat(output_list, dim=2)
        return output


class SelfMultiHeadAttentionGoogle(MultiHeadAttentionLayerGoogle):
    def forward(self, X):
        return super().forward(X, X, X)


if __name__ == '__main__':
    SEQ_LEN = 9
    BATCH_SIZE = 4
    INPUT_DIM = 8
    h = t.randn(SEQ_LEN, BATCH_SIZE, INPUT_DIM)
    Z_SIZE = 3
    Z_DIM = 5
    z = t.randn(BATCH_SIZE, Z_SIZE, Z_DIM)
    print(z.size()[2])

    model = AttentionLayerV2(INPUT_DIM, Z_SIZE, Z_DIM)
    model(h, z)