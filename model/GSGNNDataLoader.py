import dgl
import utils
import random
import dgl
import torch
from model.GSGNN import *
import os

seed = 42
# kRYST4L = 3208
ADream = 103108121

class GSGNN_DataLoader:
    def __init__(self, loader_type, random_seed, n_train, data_pth = "/home/pengxuan/Project/CSteinerPred/data", batch_size = 7):
        self.set_random_seed(random_seed)
        self.bms_pth = data_pth + "/mybenchmarks"
        self.checkpoint_dir = data_pth + "/checkpoints"
        self.batch_size = batch_size

        self.available_data = "aes_cipher_top aes128 aes192 aes256 blabla BM64 chacha des md5 ocs_blitter picorv32a salsa20 usbf_device wbqspiflash".split() # usb xtea
        self.test_data_keys = "aes_cipher_top picorv32a usbf_device wbqspiflash".split()
        self.train_data_keys = [k for k in self.available_data if k not in self.test_data_keys]
        
        self.train_set = {}
        self.test_set = {}

        self.get_train_test_dict(loader_type)

    def set_random_seed(self, random_seed):
        utils.set_random_seed(random_seed)

    def get_train_test_dict(self, model_name):
        if model_name == "GSGNN_Sub0" or model_name == "GSGNN_Sub1" or model_name == "GSGNN_Sub2":#with capacity
            for bm in self.train_data_keys:
                self.train_set[bm] = dgl.load_graphs(f"{self.bms_pth}/{bm}/gsgnn_NND_{bm}.bin")[0][0].to('cuda')
            for bm in self.test_data_keys:
                self.test_set[bm] = dgl.load_graphs(f"{self.bms_pth}/{bm}/gsgnn_NND_{bm}.bin")[0][0].to('cuda')
        elif model_name == "GSGNN_Sub3":
            for bm in self.train_data_keys:
                self.train_set[bm] = dgl.load_graphs(f"{self.bms_pth}/{bm}/gsgnn_END_{bm}.bin")[0][0].to('cuda')
            for bm in self.test_data_keys:
                self.test_set[bm] = dgl.load_graphs(f"{self.bms_pth}/{bm}/gsgnn_END_{bm}.bin")[0][0].to('cuda')
        elif model_name == "GSGNN_Sub4" or model_name == "GSGNN_Sub5" or model_name == "GSGNN":
            for bm in self.train_data_keys:
                self.train_set[bm] = dgl.load_graphs(f"{self.bms_pth}/{bm}/gsgnn_CAND_{bm}.bin")[0][0].to('cuda')
            for bm in self.test_data_keys:
                self.test_set[bm] = dgl.load_graphs(f"{self.bms_pth}/{bm}/gsgnn_CAND_{bm}.bin")[0][0].to('cuda')

    def load_model(self, model_name, tag = 0, load_path = None):
        self.checkpoint_dir = f"{self.checkpoint_dir}/{model_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if model_name == "GSGNN_Sub0":
            model = GSGNN_Sub0().to('cuda')
        elif model_name == "GSGNN_Sub1":
            model = GSGNN_Sub1().to('cuda')
        elif model_name == "GSGNN_Sub2":
            model = GSGNN_Sub2().to('cuda')
        elif model_name == "GSGNN_Sub3":
            model = GSGNN_Sub3().to('cuda')
        elif model_name == "GSGNN_Sub4":
            model = GSGNN_Sub4().to('cuda')
        elif model_name == "GSGNN_Sub5":
            model = GSGNN_Sub5().to('cuda')
        elif model_name == "GSGNN":
            model = GSGNN().to('cuda')
    


        if load_path is not None:
            model.load_state_dict(torch.load(load_path))
        else:
            load_path = f"{self.checkpoint_dir}/{tag-1}.pth"
            try:
                model.load_state_dict(torch.load(load_path))
            except:
                print(f"No checkpoint found at specified path. Start training without checkpoint")

        return model
