import utils
import random
import torch
import os

kRYST4L = 3208
ADream = 103108121
utils.set_random_seed(ADream)

# 14个benchmark
available_data = "aes_cipher_top aes128 aes192 aes256 blabla BM64 chacha des md5 ocs_blitter picorv32a salsa20 usbf_device wbqspiflash".split()

# 随机划分训练集和测试集
train_data_keys = random.sample(available_data, 10)
test_data_keys = [k for k in available_data if k not in train_data_keys]

train_set = {}
test_set = {}

abs_data_path = "/home/pengxuan/Project/CSteinerPred/data"
bms_pth = f"{abs_data_path}/mybenchmarks"

# 加载训练数据
for bm in train_data_keys:
    # 加载已经处理好的图像格式数据 (C, H, W)
    features = torch.load(f"{bms_pth}/{bm}/{bm}_gn_comb_features.pt").to('cuda')
    labels = torch.load(f"{bms_pth}/{bm}/{bm}_gn_comb_labels.pt").to('cuda')
    
    train_set[bm] = {
        "features": features,  # (C, H, W)
        "labels": labels.view(-1, 1)  # (H*W, 1)
    }

# 加载测试数据
for bm in test_data_keys:
    features = torch.load(f"{bms_pth}/{bm}/{bm}_gn_comb_features.pt").to('cuda')
    labels = torch.load(f"{bms_pth}/{bm}/{bm}_gn_comb_labels.pt").to('cuda')
    
    test_set[bm] = {
        "features": features,  # (C, H, W)
        "labels": labels.view(-1, 1)  # (H*W, 1)
    }

print("Training benchmarks:", train_data_keys)
print("Testing benchmarks:", test_data_keys)
