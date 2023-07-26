import yaml

import torch
import torch.nn as nn

with open('./test_torch/test.yaml', 'r') as test_yaml:
    config = yaml.safe_load(test_yaml)

layer_numbers = config['model_config']['layer_numbers']
# print("layer_numbers : ", layer_numbers)

def create_model():

# モデルの層を格納する空のリスト
    layers = []

    # layerの各要素を繰り返し処理
    for layer_number, layer_data in config['layer'].items():
        if layer_data[0][0] == 'conv':
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups = layer_data[1]
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
            layers.append(conv)
            # print(f"Layer {layer_number}: nn.Conv2dに対応する処理:", conv)
        elif layer_data[0][0] == 'relu':
            relu = nn.ReLU()
            layers.append(relu)
            # print(f"Layer {layer_number}: nn.ReLUに対応する処理:", relu)

    layers.append(nn.MaxPool2d(4))
    layers.append(nn.Flatten())  # 2次元画像を1次元に変換
    layers.append(nn.Linear(8 * 7 * 7, 10))  # 7x7の特徴マップから10クラスの出力を得る

    # layersに格納されたモデル層を使ってモデルを構築
    model = nn.Sequential(*layers)
    print(model)

    return model
