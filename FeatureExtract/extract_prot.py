from time import sleep

import torch
from transformers import T5EncoderModel, T5Tokenizer

import re
import numpy as np
import gc
import os
import pandas as pd

def read_data_file_trip(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results=[]
    block=len(data)//2
    for index in range(block):
        item1=data[index*2+0].split()
        name =item1[0].strip()
        seq=item1[1].strip()
        item2 = data[index * 2 + 1].split()
        item = []
        item.append(name)
        item.append(seq)
        item.append(len(seq))
        item.append(item2[1].strip())
        results.append(item)
    return results

def read_data_file_trip_from_fasta(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results=[]
    block=len(data)//2
    for index in range(block):
        name=data[index*2+0].replace('>','').strip()
        seq = data[index * 2 + 1].strip()
        item = []
        item.append(name)
        item.append(seq)
        results.append(item)
    return results
def extratdata(file,destfolder):
    student_tuples=read_data_file_trip(file)
    #student_tuples = sorted(student_tuples, key=lambda student: student[2], reverse=True)
    # with open(os.path.join(destfolder, 'file_prot.txt'), 'w') as f:
    #     f.write('\n'.join(item[0] for item in student_tuples))
    i=1
    recordlength=len(student_tuples)
    for name, seq, length, label in student_tuples:
        print(f'progress {i}/{recordlength}')
        i += 1
        if os.path.exists(os.path.join(destfolder, name + '.data')):
            print(name,' existed')
            continue

        with open(os.path.join(destfolder, name + '.label'), 'w') as f:
            f.write(','.join(l for l in label))
        newseq=' '.join(s for s in seq)
        newseq=re.sub(r"[UZOB]", "X", newseq)
        #print('newseq length',len(newseq))
        ids = tokenizer.batch_encode_plus([newseq], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            #features.append(seq_emd)
            with open(os.path.join(destfolder, name + '.data'), 'w') as f:
                np.savetxt(f, seq_emd, delimiter=',', fmt='%s')

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        # sleep(0.5)
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        #model.to('cpu')

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:1'
    model = model.to(device)
    model = model.eval()

    # print('----prepare traindataset-------')
    extratdata('../DataSet/Train/Nuc-1521_train_all.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/Train/Nuc-798_Train_all.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/Train/Nuc-849_Train_all.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/Train/Nuc_207_train_all.txt', '../DataSet/prot_embedding/')
    # print('----prepare testdataset-------')
    extratdata('../DataSet/Nuc-1521_Test_all.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/Nuc-798_Test_all.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/Nuc-849_Test_all.txt', '../DataSet/prot_embedding/')
    extratdata('../DataSet/Nuc_207_Test_all.txt', '../DataSet/prot_embedding/')
    print('----finish-------')
