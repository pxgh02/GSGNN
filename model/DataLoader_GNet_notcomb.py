import dgl
import utils
import random
import torch
from data.data_type import getGR_WL



utils.set_random_seed(1)

#14
available_data = "aes_cipher_top aes128 aes192 aes256 blabla BM64 chacha des md5 ocs_blitter picorv32a salsa20 usbf_device wbqspiflash".split() # usb xtea


train_data_keys = random.sample(available_data, 10)
test_data_keys = [k for k in available_data if k not in train_data_keys]

train_set = {}
test_set = {}

abs_data_path = "/home/pengxuan/Project/CSteinerPred/data"
bms_pth = f"{abs_data_path}/mybenchmarks"


# 加载训练数据
for bm in train_data_keys:
    g = dgl.load_graphs(f"{bms_pth}/{bm}/{bm}_gn.bin")[0][0].to('cuda')
    WL = torch.tensor(getGR_WL(f"{bms_pth}/{bm}/gr_result.log"), dtype=torch.float32).to('cuda')
    dict_bm = {}
    dict_bm["graph"] = g
    dict_bm["WL"] = WL
    train_set[bm] = dict_bm



# 加载测试数据
for bm in test_data_keys:
    g = dgl.load_graphs(f"{bms_pth}/{bm}/{bm}_gn.bin")[0][0].to('cuda')
    WL = torch.tensor(getGR_WL(f"{bms_pth}/{bm}/gr_result.log"), dtype=torch.float32).to('cuda')
    dict_bm = {}
    dict_bm["graph"] = g
    dict_bm["WL"] = WL
    test_set[bm] = dict_bm

print(train_data_keys)
print(test_data_keys)
