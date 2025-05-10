import dgl
import utils
import random
import dgl
import torch
from model.stgsg_v3 import *
from model.stgsg import *
from model.gscgnn import *
import os

# kRYST4L = 3208
ADream = 103108121

class STGSG_DataLoader:
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

    # def split_train_test(self, n_train):
    #     train_data_keys = random.sample(self.available_data, n_train)
    #     test_data_keys = [k for k in self.available_data if k not in train_data_keys]
    #     print(train_data_keys)
    #     print(test_data_keys)
    #     return train_data_keys, test_data_keys

    def get_train_test_dict(self, loader_type):
        if loader_type == "STGSG_V3" or loader_type == "STGSG_V4" or loader_type == "STGSG_V4d1":#with capacity
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/{bm}_gs_v3.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/{bm}_gs_v3.bin")[0][0].to('cuda')
                self.test_set[bm] = g
        
        elif loader_type == "GSCGnn_V1" or loader_type == "GSCGnn_V2" or loader_type == "GSCGnn_V3" or loader_type == "GSCGnn_V4":
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}.bin")[0][0].to('cuda')
                self.test_set[bm] = g
        elif loader_type == "GSCGnn_V5":
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v2.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v2.bin")[0][0].to('cuda')
                self.test_set[bm] = g

        elif loader_type == "GSCGnn_V6":
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v3.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v3.bin")[0][0].to('cuda')
                self.test_set[bm] = g
        elif loader_type == "GSCGnn_V7":
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v4.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v4.bin")[0][0].to('cuda')
                self.test_set[bm] = g

        elif loader_type == "GSCGnn_V8":
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v5.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v5.bin")[0][0].to('cuda')
                self.test_set[bm] = g   

        elif loader_type == "GSCGnn_V9" or loader_type == "GSCGnn_V10" or loader_type == "GSCGnn_V11":
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v6.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v6.bin")[0][0].to('cuda')
                self.test_set[bm] = g
        
        elif loader_type == "GSCGnn_V10" or loader_type == "GSCGnn_V12":
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v7.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/gscgnn_{bm}_v7.bin")[0][0].to('cuda')
                self.test_set[bm] = g

        elif loader_type == "STGSG2d1" or loader_type == "STGSG2d1_NO_VS" or loader_type == "STGSG2d1_NO_VSVCPIN" or loader_type == "STGSG2d1_NO_VC_PINDENSITY" or loader_type == "STGSG2d1_NO_VC_NETDENSITY" or loader_type == "STGSG2d1_ONLY_VC_NETDENSITY" or loader_type == "STGSG2d1_ONLY_VC_PINPINDensity":#comb_H/V
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/{bm}_gs_v2d1.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/{bm}_gs_v2d1.bin")[0][0].to('cuda')
                self.test_set[bm] = g
        

                
        elif loader_type == "STGSG2d5":#not comb_H/V
            for bm in self.train_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/{bm}_gs_v2d5.bin")[0][0].to('cuda')
                self.train_set[bm] = g
            for bm in self.test_data_keys:
                g = dgl.load_graphs(f"{self.bms_pth}/{bm}/{bm}_gs_v2d5.bin")[0][0].to('cuda')
                self.test_set[bm] = g

        else:
            print(f"loader_type {loader_type} is not supported!")
            exit()

    def load_model(self, model_name, tag = 0, load_path = None):
        self.checkpoint_dir = f"{self.checkpoint_dir}/{model_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if model_name == "STGSG_V3" or model_name == "STGSG_V4" or model_name == "STGSG_V4d1":
            model = STGSG_V3(5, 1).to('cuda')
        
        elif model_name == "GSCGnn_V1":
            model = GSCGNN_V1().to('cuda')

        elif model_name == "GSCGnn_V2":
            model = GSCGNN_V2().to('cuda')

        elif model_name == "GSCGnn_V3":
            model = GSCGNN_V3().to('cuda')

        elif model_name == "GSCGnn_V4":
            model = GSCGNN_V4().to('cuda')
        
        elif model_name == "GSCGnn_V5":
            model = GSCGNN_V5().to('cuda')

        elif model_name == "GSCGnn_V6":
            model = GSCGNN_V6().to('cuda')

        elif model_name == "GSCGnn_V7":
            model = GSCGNN_V7().to('cuda')

        elif model_name == "GSCGnn_V8":
            model = GSCGNN_V8().to('cuda')

        elif model_name == "GSCGnn_V9":
            model = GSCGNN_V9().to('cuda')

        elif model_name == "GSCGnn_V10":
            model = GSCGNN_V10().to('cuda')

        elif model_name == "GSCGnn_V11":
            model = GSCGNN_V11().to('cuda')

        elif model_name == "GSCGnn_V12":
            model = GSCGNN_V12().to('cuda')

        elif model_name == "STGSG2d1":
            model = STGSG2d1(5, 5).to('cuda')
        
        elif model_name == "STGSG2d1_NO_VS":
            model = STGSG2d1_NO_VS(5, 5).to('cuda')

        elif model_name == "STGSG2d1_NO_VSVCPIN":
            model = STGSG2d1_NO_VSVCPIN(5, 5).to('cuda')

        elif model_name == "STGSG2d1_NO_VC_PINDENSITY":
            model = STGSG2d1_NO_VC_PINDENSITY(5, 5).to('cuda')

        elif model_name == "STGSG2d1_NO_VC_NETDENSITY":
            model = STGSG2d1_NO_VC_NETDENSITY(5, 5).to('cuda')

        elif model_name == "STGSG2d1_ONLY_VC_NETDENSITY":
            model = STGSG2d1_ONLY_VC_NETDENSITY(5, 5).to('cuda')

        elif model_name == "STGSG2d1_ONLY_VC_PINPINDensity":
            model = STGSG2d1_ONLY_VC_PINPINDensity(5, 5).to('cuda')

        elif model_name == "STGSG2d5":
            model = STGSG2d5(5, 5).to('cuda')

        if load_path is not None:
            model.load_state_dict(torch.load(load_path))
        else:
            load_path = f"{self.checkpoint_dir}/{tag-1}.pth"
            try:
                model.load_state_dict(torch.load(load_path))
            except:
                print(f"No checkpoint found at specified path. Start training without checkpoint")

        return model
