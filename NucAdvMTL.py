import datetime


import numpy as np
import pandas as pd
import torch
import  torch.nn as nn
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from LossFunction.focalLoss import FocalLoss_v2
import torch.multiprocessing

import datetime



def read_data_file_trip(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results=[]
    block=len(data)//2
    for index in range(block):
        item=data[index*2+0].split()
        name =item[0].strip()
        results.append(name)
    return results

def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature_plms = []
    features_proteins=[]
    train_y = []
    task_ids=[]


    for data in batch_traindata:
        feature_plms.append(data[0])
        features_proteins.append(data[1])
        train_y.append(data[2])
        task_ids.append(data[3])
    data_length = [len(data) for data in feature_plms]

    feature_plms = torch.nn.utils.rnn.pad_sequence(feature_plms, batch_first=True, padding_value=0)
    features_proteins = torch.nn.utils.rnn.pad_sequence(features_proteins, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    task_ids = torch.nn.utils.rnn.pad_sequence(task_ids, batch_first=True, padding_value=0)
    return feature_plms,features_proteins,train_y,task_ids,torch.tensor(data_length)
class GRLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lmbd=0.01):
        ctx.lmbd = torch.tensor(lmbd)
        return x.reshape_as(x)

    @staticmethod
    # 输入为forward输出的梯度
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.lmbd * grad_input.neg(), None


class BioinformaticsDataset(Dataset):
    # X: list of filename
    def __init__(self, X):
        self.X = X
    def __getitem__(self, index):
        filename = self.X[index]
        #esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        df0 = pd.read_csv('DataSet/prot_embedding/' + filename + '.data', header=None)
        prot = df0.values.astype(float).tolist()
        prot = torch.tensor(prot)

        agv_pro = torch.mean(prot, dim=0)
        agv_pro = agv_pro.repeat(prot.shape[0], 1)

        agv_res = torch.mean(prot, dim=1)
        agv_res = agv_res.unsqueeze(dim=1)

        agv_pro=torch.cat((agv_pro,agv_res),dim=1)

        df2= pd.read_csv('DataSet/prot_embedding/'+  filename+'.label', header=None)
        label = df2.values.astype(int).tolist()
        label = torch.tensor(label)
        #reduce 2D to 1D
        label=torch.squeeze(label)
        #ADP-0; AMP-1; ATP-2; GDP-3; GTP-4
        if '_ADP' in filename:
            task_id_label=torch.ones(prot.shape[0],dtype=int)*0
        elif '_AMP' in filename:
            task_id_label=torch.ones(prot.shape[0],dtype=int)*1
        elif '_GDP' in filename:
            task_id_label=torch.ones(prot.shape[0],dtype=int)*3
        elif '_GTP' in filename:
            task_id_label=torch.ones(prot.shape[0],dtype=int)*4
        else:  # ATP
            task_id_label=torch.ones(prot.shape[0],dtype=int)*2

        return prot,agv_pro, label,task_id_label


    def __len__(self):
        return len(self.X)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.discriminator_block_fc = nn.Sequential(nn.Linear(128, 512),
                                           nn.Dropout(0.5),
                                           nn.Linear(512, 64),
                                           nn.Dropout(0.5),
                                           nn.Linear(64, 5))
    def forward(self,embedding):
        return self.discriminator_block_fc(embedding)

class AttentionModel(nn.Module):
    def __init__(self, q_inutdim, k_inputdim, v_inutdim):
        super(AttentionModel, self).__init__()
        self.q = nn.Linear(q_inutdim, q_inutdim)
        self.k = nn.Linear(k_inputdim, k_inputdim)
        self.v = nn.Linear(v_inutdim, v_inutdim)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(k_inputdim))

    def forward(self, plms1, plms2,plms3, seqlengths):
        Q = self.q(plms1)
        K = self.k(plms2)
        V = self.v(plms3)
        atten=self.masked_softmax((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact,seqlengths)
        output = torch.bmm(atten, V)
        return output + plms3
    def create_src_lengths_mask(self, batch_size: int, src_lengths):
        max_src_len = int(src_lengths.max())
        src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
        src_indices = src_indices.expand(batch_size, max_src_len)
        src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
        # returns [batch_size, max_seq_len]
        return (src_indices < src_lengths).int().detach()

    def masked_softmax(self, scores, src_lengths, src_length_masking=True):
        # scores [batchsize,L*L]
        if src_length_masking:
            bsz, src_len, max_src_len = scores.size()
            # compute masks
            src_mask = self.create_src_lengths_mask(bsz, src_lengths)
            src_mask = torch.unsqueeze(src_mask, 2)
            # Fill pad positions with -inf
            scores = scores.permute(0, 2, 1)
            scores = scores.masked_fill(src_mask == 0, -np.inf)
            scores = scores.permute(0, 2, 1)
        return F.softmax(scores.float(), dim=-1)

class Task_shared(nn.Module):
    def __init__(self,inputdim):
        super(Task_shared,self).__init__()
        self.inputdim=inputdim
        self.input_block = nn.Sequential(
            nn.LayerNorm(self.inputdim, eps=1e-6)
            , nn.Linear(self.inputdim, 128)
            , nn.LeakyReLU()
        )

        self.hidden_block = nn.Sequential(
            nn.LayerNorm(128, eps=1e-6)
            , nn.Dropout(0.2)
            , nn.Linear(128, 128)
            , nn.LeakyReLU()
            , nn.LayerNorm(128, eps=1e-6)
        )

        self.share_task_block1 = nn.Sequential(nn.Conv1d(self.inputdim, 512, 1, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(512, 256, 1, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(256, 128, 1, padding='same'),
                                               nn.ReLU(True))
        self.share_task_block2 = nn.Sequential(nn.Conv1d(self.inputdim, 512, 3, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(512, 256, 3, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(256, 128, 3, padding='same'),
                                               nn.ReLU(True))
        self.share_task_block3 = nn.Sequential(nn.Conv1d(self.inputdim, 512, 5, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(512, 256, 5, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(256, 128, 5, padding='same'),
                                               nn.ReLU(True))


    def forward(self,prot_input,protein_emb,datalengths):
        #Golbal
        # pl = self.input_block(prot_input)
        # pl = self.hidden_block(pl)
        # x = torch.nn.utils.rnn.pack_padded_sequence(pl, datalengths.to('cpu'), batch_first=True)
        # x, (h_n, h_c) = self.lstm(x)
        # protlstm, output_lens = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        #local
        prot_input_share = prot_input.permute(0, 2, 1)
        prot_input_share1 = self.share_task_block1(prot_input_share)
        prot_input_share2 = self.share_task_block2(prot_input_share)
        prot_input_share3 = self.share_task_block3(prot_input_share)
        prot_input_share = prot_input_share1 + prot_input_share2 + prot_input_share3
        prot_input_share=prot_input_share.permute(0,2,1)
        # prot_input_share4 = self.share_task_block4(prot_input_share)
        # prot_input_share5 = self.share_task_block5(prot_input_share)
        # prot_input_share6 = self.share_task_block6(prot_input_share)
        #
        # prot_input_share4 = prot_input_share4.permute(0, 2, 1)
        # prot_input_share5 = prot_input_share5.permute(0, 2, 1)
        # prot_input_share6 = prot_input_share6.permute(0, 2, 1)
        # prot_input_share=prot_input_share4+prot_input_share5+prot_input_share6

        return prot_input_share


class Task_SP(nn.Module):
    def __init__(self):
        super(Task_SP, self).__init__()

        self.input_block = nn.Sequential(
            nn.LayerNorm(1025, eps=1e-6)
            , nn.Linear(1025, 128)
            , nn.LeakyReLU()
        )

        self.hidden_block = nn.Sequential(
            nn.LayerNorm(128, eps=1e-6)
            , nn.Dropout(0.2)
            , nn.Linear(128, 128)
            , nn.LeakyReLU()
            , nn.LayerNorm(128, eps=1e-6)
        )

    def forward(self, protein_emb):
        # Golbal
        pl = self.input_block(protein_emb)
        pl = self.hidden_block(pl)

        return pl

class MTLModule(nn.Module):
    def __init__(self,inputdim,istrain):
        super(MTLModule,self).__init__()
        self.istrain=istrain
        self.inputdim = inputdim
        self.ShardEncoder=Task_shared(self.inputdim)
        self.discriminator1=  Discriminator()

        #ADP-0; AMP-1; ATP-2; GDP-3; GTP-4
        # self.tasks_encoders = nn.ModuleList()
        self.privates = nn.ModuleList()
        self.tasks_fcs = nn.ModuleList()
        self.attns=nn.ModuleList()
        # for i in range(5):
        #     self.tasks_encoders.append(Task_Model())

        for i in range(5):
            self.privates.append(Task_SP())

            self.tasks_fcs.append(nn.Sequential(nn.Linear(128, 512),
                      nn.Dropout(0.5),
                      nn.Linear(512, 64),
                      nn.Dropout(0.5),
                      nn.Linear(64, 2)))
    def diff_loss(self,shared_embeding1,shared_embeding2):
        innermul=shared_embeding1*shared_embeding2
        l2=torch.norm(innermul,dim=2)
        return torch.sum(l2)
        # share_h = F.normalize(shared_embeding - torch.mean(shared_embeding, 0))
        # task_h = F.normalize(task_embedding - torch.mean(task_embedding, 0))
        # dot_mat = torch.matmul(share_h, task_h.transpose(0, 1))
        # return torch.sum(dot_mat ** 2)

    def forward(self,prot_input,protein_emb,datalengths):
        sharedembedding=self.ShardEncoder(prot_input,protein_emb,datalengths)

        task_outs=[]
        #task_embedding=self.attention(prot_input_share1, prot_input_share3, prot_input_share5,datalengths)
        #task_embedding = prot_input_share1+ prot_input_share3+ prot_input_share5
        for i in range(5):
            task_embedding_private= self.privates[i](protein_emb)
            task_embeddingi=self.tasks_fcs[i](sharedembedding+task_embedding_private)
            task_outs.append(task_embeddingi)
        prot_input_share= GRLayer.apply(sharedembedding)
        discriminator1=self.discriminator1(prot_input_share)

        return task_outs,discriminator1

def train(itrainfile,modelstoreapl):
    model = MTLModule(1024,True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30

    per_cls_weights = torch.FloatTensor([0.15, 0.85]).to(device)
    #per_cls_weights = torch.FloatTensor([0.25, 0.75]).to(device)
    fcloss = FocalLoss_v2(alpha=per_cls_weights, gamma=2)
    discrossloss = nn.CrossEntropyLoss()
    model.train()

    file = read_data_file_trip(itrainfile)
    train_set = BioinformaticsDataset(file)
    train_loader = DataLoader(dataset=train_set, batch_size=16,pin_memory=True,
                              persistent_workers=True,shuffle=True, num_workers=16,
                              collate_fn=coll_paddding)
    best_val_loss = 3000
    best_epo=0
    for epo in range(epochs):
        epoch_loss_train = 0
        nb_train = 0
        for prot_xs,p_emb, data_ys, taskids, lengths in train_loader:
            task_outs,discriminator = model(prot_xs.to(device),p_emb.to(device), lengths.to(device))
            data_ys = data_ys.to(device)
            #print('data_ys shape,', data_ys.shape)
            taskids=taskids.to(device)
            lengths=lengths.to('cpu')
            for i in range(5):
                task_outs[i] = torch.nn.utils.rnn.pack_padded_sequence(task_outs[i], lengths, batch_first=True)
            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)
            #print('data_ys.data shape,', data_ys.data.shape)
            discriminator= torch.nn.utils.rnn.pack_padded_sequence(discriminator, lengths, batch_first=True)

            taskids = torch.nn.utils.rnn.pack_padded_sequence(taskids, lengths, batch_first=True)
            loss_share = discrossloss(discriminator.data, taskids.data)

            loss_task=0
            for i in range(5):
                indexs = torch.nonzero(taskids.data == i).squeeze()
                pred=task_outs[i].data[indexs]
                lbs=data_ys.data[indexs]
                if lbs.shape[0]>0:
                    fc = fcloss(pred, lbs)
                    loss_task += fc
            lose = loss_task+loss_share*0.5 # +loss_share
            #lose = loss_task+ (loss_share1+loss_share3+loss_share5)*0.1

            optimizer.zero_grad()
            lose.backward()
            # clip_grad_norm_要放在backward和step之间
            optimizer.step()
            epoch_loss_train = epoch_loss_train + lose.item()
            nb_train += 1
        epoch_loss_avg = epoch_loss_train / nb_train
        print('epo ',epo,' epoch_loss_avg,', epoch_loss_avg)
        if best_val_loss > epoch_loss_avg:
            model_fn = modelstoreapl
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            best_epo=epo
            if epo % 10 == 0:
                print('epo ',epo," Save model, best_val_loss: ", best_val_loss)
    print('best loss,',best_val_loss,'best epo,',best_epo)

def test(itestfile,modelstoreapl):
    model = MTLModule(1024,False)
    model = model.to(device)
    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()
    tmresult = {}

    file = read_data_file_trip(itestfile)
    test_set = BioinformaticsDataset(file)
    test_load = DataLoader(dataset=test_set, batch_size=32,
                           num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)

    print("==========================Test RESULT================================")

    predicted_probs = [[],[],[],[],[]]
    labels_actual = [[],[],[],[],[]]
    labels_predicted = [[],[],[],[],[]]

    with torch.no_grad():
        for prot_xs,pro_emb,data_ys,taskids ,lengths in  test_load:
            task_outs,discriminator = model(prot_xs.to(device),pro_emb.to(device), lengths.to(device))
            for i in range(5):
                task_outs[i] = torch.nn.utils.rnn.pack_padded_sequence(task_outs[i], lengths.to('cpu'),
                                                                       batch_first=True)

            data_ys = torch.nn.utils.rnn.pack_padded_sequence(data_ys, lengths, batch_first=True)
            taskids = torch.nn.utils.rnn.pack_padded_sequence(taskids, lengths, batch_first=True)
            # ADP-0; AMP-1; ATP-2; GDP-3; GTP-4
            for i in range(5):
                indexs = torch.nonzero(taskids.data == i).squeeze()
                task_pred = task_outs[i].data[indexs]
                lbs = data_ys.data[indexs]
                task_pred = torch.nn.functional.softmax(task_pred, dim=1)
                task_pred = task_pred.to('cpu')
                if lbs.shape[0]>0:
                    predicted_probs[i].extend(task_pred[:, 1])
                    labels_actual[i].extend(lbs)
                    labels_predicted[i].extend(torch.argmax(task_pred, dim=1))

        itask_names=['ADP', 'AMP', 'ATP', 'GDP','GTP']
        itaskid=[0,1,2,3,4]
        for id,task_name in zip(itaskid,itask_names):
            sensitivity, specificity, acc, precision, mcc, auc, aupr_1 = printresult(task_name, labels_actual[id],
                                                                                 predicted_probs[id],
                                                                                 labels_predicted[id])
            tmresult[task_name] = [sensitivity, specificity, acc, precision, mcc, auc, aupr_1]
    return tmresult


def printresult(ligand,actual_label,predict_prob,predict_label):
    print('\n---------',ligand,'-------------')
    auc = metrics.roc_auc_score(actual_label, predict_prob)
    precision_1, recall_1, threshold_1 = metrics.precision_recall_curve(actual_label, predict_prob)
    aupr_1 = metrics.auc(recall_1, precision_1)
    acc=metrics.accuracy_score(actual_label, predict_label)
    print('acc ',acc )
    print('balanced_accuracy ', metrics.balanced_accuracy_score(actual_label, predict_label))
    tn, fp, fn, tp = metrics.confusion_matrix(actual_label, predict_label).ravel()
    print('tn, fp, fn, tp ', tn, fp, fn, tp)
    mcc=metrics.matthews_corrcoef(actual_label, predict_label)
    print('MCC ', mcc)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1score = 2 * precision * recall / (precision + recall)
    youden = sensitivity + specificity - 1
    print('sensitivity ', sensitivity)
    print('specificity ', specificity)
    print('precision ', precision)
    print('recall ', recall)
    print('f1score ', f1score)
    print('youden ', youden)
    print('auc', auc)
    print('AUPR ', aupr_1)
    print('---------------END------------')
    return sensitivity, specificity, acc, precision, mcc, auc, aupr_1

if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy('file_system')
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(1)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")


    train_taskfile = 'DataSet/Train/pretrain_train_all_newdataset.txt'
    #train_taskfile = 'DataSet/Train/pretrain_train_all227.txt'
    #train_taskfile = 'DataSet/Train/pretrain_train_all221.txt'

    test_taskfie = 'DataSet/Test/pretrain_test_all_newdataset.txt'
    #test_taskfie = 'DataSet/Test/pretrain_test_all_17.txt'
    #test_taskfie = 'DataSet/Test/pretrain_test_all_50.txt'

    circle=5
    a = str(datetime.datetime.now())
    a=a.replace(':','_')

    totalkv = {'ATP': [], 'ADP': [], 'AMP': [], 'GDP': [], 'GTP': []}
    for i in range(circle):
        storeapl = 'AdvMTL/Result_ADV_T5_Nuc1_Protein_' + str(i) + '_' + a + '.pkl'
        train(train_taskfile,storeapl)
        tmresult = test(test_taskfie,storeapl)

        totalkv['ATP'].append(tmresult['ATP'])
        totalkv['ADP'].append(tmresult['ADP'])
        totalkv['AMP'].append(tmresult['AMP'])
        totalkv['GDP'].append(tmresult['GDP'])
        totalkv['GTP'].append(tmresult['GTP'])
        torch.cuda.empty_cache()

    with open('AdvMTL/Result_ADV_T5_Nuc1_Protein_' + a + '.txt', 'w') as f:
        nucs = ['ATP', 'ADP', 'AMP', 'GDP', 'GTP']
        for nuc in nucs:
            np.savetxt(f, totalkv[nuc], delimiter=',', footer='Above is  record ' + nuc, fmt='%s')
            m = np.mean(totalkv[nuc], axis=0)
            np.savetxt(f, [m], delimiter=',', footer='----------Above is AVG -------' + nuc, fmt='%s')

