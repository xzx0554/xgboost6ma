

import sys
import os


import pandas as pd
import torch
from Bio import SeqIO
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import argparse
from gensim.models import word2vec
from Bert_network import BERT
import pickle
from tqdm import tqdm
common_path = os.path.abspath("..")

def import_fasta(filename):

    return pd.read_csv(filename, sep='\t', header=0)

def pickle_save(filename, data):
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)

def emb_seq(seq, w2v_model, features, num = 4):
    seq_emb = []
    for i in range(len(seq) - num + 1):
        kmer = seq[i:i+num]
        if kmer in w2v_model.wv:
            seq_emb.append(np.array(w2v_model.wv[kmer]))
        else:
            seq_emb.append(np.zeros(features))
    return np.array(seq_emb)

class pv_data_sets(data.Dataset):
    #def __init__(self, data_sets):
    def __init__(self, data_sets, w2v_model, features, device):
        super().__init__()
        self.w2v_model = w2v_model
        self.seq = data_sets["text"].values.tolist()
        self.id = data_sets["label"].values.tolist()
        self.device = device
        self.features = features

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        emb_mat = emb_seq(self.seq[idx], self.w2v_model, self.features)
        
        return torch.tensor(emb_mat).to(self.device).float(), self.id[idx]

def output_csv_pandas(filename, data):
    data.to_csv(filename, index = None)

class burt_process():
    def __init__(self, out_path, deep_model_path, batch_size = 64, features = 100, thresh = 0.5,pickle_name = 'A.thaliana',test_or_train ='train',dataset_name = 'A.thaliana'):
        self.out_path = out_path
        self.deep_model_path = deep_model_path
        self.batch_size = batch_size
        self.features = features
        self.thresh = thresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pickle_name = pickle_name
        self.test_or_train = test_or_train
        self.dataset_name = dataset_name
    def pre_training(self, dataset, w2v_model):
        os.makedirs(self.out_path, exist_ok = True) 
        data_all = pv_data_sets(dataset, w2v_model, self.features, self.device)
        loader = DataLoader(dataset = data_all, batch_size = self.batch_size, shuffle=False)
        
        net = BERT(n_layers = 3, d_model = self.features, n_heads = 4, d_dim = 100, d_ff = 400, time_seq = 41 - 4 + 1).to(self.device)
        net.load_state_dict(torch.load(self.deep_model_path, map_location = self.device))
            
        print("The number of data:" + str(len(dataset)))
            
        probs, pred_labels, seq_id_list, att_w_1, att_w_2, att_w_3 = [], [], [], [], [], []
        output_dict = {}  # 创建一个空字典来存储输出和序列ID
      
        print("predicting...")
        net.eval()
        for i, (emb_mat, seq_id) in tqdm(enumerate(loader)):
            with torch.no_grad():
                outputs,output_my = net(emb_mat)
                        
            probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
            pred_labels.extend((np.array(outputs.cpu().detach().squeeze(1).numpy()) + 1 - self.thresh).astype(np.int16))        
            seq_id_list.extend(seq_id)
            for id, output in zip(seq_id, output_my):
                output_dict[i] = {"label":id,"tensor":output.cpu().numpy()}
            att_w_1.extend(net.attn_list[0].cpu().detach().numpy()) 
            att_w_2.extend(net.attn_list[1].cpu().detach().numpy()) 
            att_w_3.extend(net.attn_list[2].cpu().detach().numpy()) 
            
        print("finished the prediction")

        print("saving results...")
        res = pd.DataFrame([seq_id_list, probs, pred_labels]).transpose()
        res.columns = ["id", "probability", "predictive labels"]
        output_csv_pandas(self.out_path + "/results.csv", res)
        att_weights = np.transpose(np.array([att_w_1, att_w_2, att_w_3]), (1, 0, 2, 3, 4))
        pickle_save(self.out_path + "/attention_weights.pkl", np.array(att_weights))
        pickle_save(self.out_path + f"/model_{self.pickle_name}_dataset_{self.dataset_name}_{self.test_or_train }.pkl", output_dict)

        print("finished all processes")


choices=['C.elegans', 'C.equisetifolia', 'D.melanogaster', 'F.vesca', 'H.sapiens', 'R.chinensis', 'S.cerevisiae', 'T.thermophile', 'Tolypocladium', 'Xoc.BLS256']

train_or_test_choices = ['train','test']
for i in tqdm(choices):
        for name in choices:
            if name == i :
               for b in train_or_test_choices:
                    test_path = "../dataset/6mA_"+name+'/'+b+'.tsv'
                    out_path ="./"
                    w2v_model = word2vec.Word2Vec.load(common_path + "/w2v_model/dna_w2v_100.pt")
                    dataset = import_fasta(test_path)

                    net = burt_process(out_path, deep_model_path = common_path + "/deep_model/6mA_" + i + "/deep_model", batch_size =1, thresh = float(0.5),pickle_name =i,dataset_name =name,test_or_train = b)
                    net.pre_training(dataset, w2v_model)
            else:
                b=train_or_test_choices[1]
            test_path = "../dataset/6mA_"+name+'/'+b+'.tsv'
            out_path ="./"
            w2v_model = word2vec.Word2Vec.load(common_path + "/w2v_model/dna_w2v_100.pt")
            dataset = import_fasta(test_path)

            net = burt_process(out_path, deep_model_path = common_path + "/deep_model/6mA_" + i + "/deep_model", batch_size =1, thresh = float(0.5),pickle_name =i,dataset_name =name,test_or_train = b)
            net.pre_training(dataset, w2v_model)