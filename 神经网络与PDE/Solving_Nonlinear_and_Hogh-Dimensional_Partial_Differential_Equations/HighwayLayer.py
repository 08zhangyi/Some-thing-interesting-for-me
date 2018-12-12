import torch as t


# 初始层，从x生成S
class HighwayLayerFirst(t.nn.Module):
    def __init__(self, input_feature_size, S_feature_size):
        super().__init__()
        self.layer = t.nn.Linear(input_feature_size, S_feature_size)

    def forward(self, x):
        S = self.layer(x)
        S = t.nn.ReLU()(S)
        return S


# 中间层，从x，S生成S
class HighwayLayerNormal(t.nn.Module):
    def __init__(self, input_feature_size, S_feature_size):
        super().__init__()
        self.layer_Z_1 = t.nn.Linear(S_feature_size, S_feature_size, bias=False)
        self.layer_Z_2 = t.nn.Linear(input_feature_size, S_feature_size)
        self.layer_G_1 = t.nn.Linear(S_feature_size, S_feature_size, bias=False)
        self.layer_G_2 = t.nn.Linear(input_feature_size, S_feature_size)
        self.layer_R_1 = t.nn.Linear(S_feature_size, S_feature_size, bias=False)
        self.layer_R_2 = t.nn.Linear(input_feature_size, S_feature_size)
        self.layer_H_1 = t.nn.Linear(S_feature_size, S_feature_size, bias=False)
        self.layer_H_2 = t.nn.Linear(input_feature_size, S_feature_size)

    def forward(self, x ,S):
        Z = t.nn.ReLU()(self.layer_Z_1(S) + self.layer_Z_2(x))
        G = t.nn.ReLU()(self.layer_G_1(S) + self.layer_G_2(x))
        R = t.nn.ReLU()(self.layer_R_1(S) + self.layer_R_2(x))
        H = t.nn.ReLU()(self.layer_H_1(S * R) + self.layer_H_2(x))
        S = (1.0 - G) * H + Z * S
        return S


# 有L层的Highway网络的实现
class HighwayNetwork(t.nn.Module):
    def __init__(self, input_feature_size, S_feature_size, L=1):
        super().__init__()
        self.layer_init = HighwayLayerFirst(input_feature_size, S_feature_size)
        self.layers = [HighwayLayerNormal(input_feature_size, S_feature_size) for _ in range(L)]
        for i, layer in enumerate(self.layers):
            setattr(self, 'highway_layer_'+str(i), layer)

    def forward(self, x):
        S = self.layer_init(x)
        for i, layer in enumerate(self.layers):
            S = layer(x, S)
        return S


if __name__ == '__main__':
    batch_size = 8
    input_feature_size = 10
    S_feature_size = 12
    x = t.randn(batch_size, input_feature_size)
    # 调用网络结构
    model = HighwayNetwork(input_feature_size, S_feature_size, 5)
    S = model(x)
    print(x.size(), S.size())
    print(list(model.parameters()))