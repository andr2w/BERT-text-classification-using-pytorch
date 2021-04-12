import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    '''
    config parameter
    '''
    def __init__(self, dataset):
        self.model_name = 'an-bert'
        # train dataset
        self.train_path = dataset + '/data/train.txt'
        # test dataset
        self.test_path = dataset + '/data/test.txt'
        # dev dataset
        self.dev_path = dataset + '/data/dev.txt'

        # label 
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # num of labels
        self.num_classes = len(self.class_list)

        # saved path
        self.save_path = dataset + '/save_dir/' + self.model_name + '.ckpt'

        # auto choose cpu or gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        # if batch > 1000, the performance hasn't become better, end ealier
        self.require_imrovement = 1000

        # num of epoches
        self.num_epochs = 3

        # batch size
        self.batch_size = 128
        ''' 
         padding size 
         the max length of each sentence
         long cut, short add 
        '''
        self.pad_size = 32
        # learning rate
        self.learning_rate = 1e-5

        # bert pre-train path
        self.bert_path = 'bert_pretrain'
        # bert tokenization
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert hidden size
        self.hidden_size = 768


class Model(nn.Module):
    
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True # little tuning bert layer

        self.fc = nn.Linear(config.hidden_size, config.num_classes) 
    
    def forward(self, x):
        '''
        x  [ids, seq_len, mask]
        '''
        context = x[0] # sentence  shape[128, 32]
        mask = x[2] # mask for padding shape[128, 32]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # pooled shape[128, 768]
        out = self.fc(pooled) # shape[128, 10]
        return out


        



    
