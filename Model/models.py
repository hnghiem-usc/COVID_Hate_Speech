import torch
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from transformers import BertForSequenceClassification, AdamW , BertConfig, BertModel
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import LineByLineTextDataset


from torch.nn import LSTM, Dropout, Linear, BCELoss, CrossEntropyLoss, Softmax, ReLU, Conv2d, MaxPool2d, LeakyReLU
from torch.nn.functional import softmax, relu, leaky_relu
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


############################## BERT BASE ##############################
class Bert_base(torch.nn.Module):
    def __init__(self, bert_type, num_labels:list, max_len, attn_yn=False, dropout_rate = 0.2):
        super(Bert_base, self).__init__()
        self.model_type = 'Bert_base'
        self.bert_type = bert_type
        if len(num_labels) == 1:
            print('Sigle-task training')
        else: 
            print('Multitask training: {} tasks '.format(len(num_labels)))

        self.num_labels = num_labels
        self.num_tasks = len(self.num_labels)
        self.train_losses = [[]] * self.num_tasks
        self.val_losses = [[]] * self.num_tasks

        self.attn_yn = attn_yn
        self.max_len = torch.tensor([max_len])
        self.dropout_rate = 0.2
        # BERT 
        self.bert = BertModel.from_pretrained(self.bert_type,
                                                   output_attentions = attn_yn, 
                                                   output_hidden_states=False
                                              )
        self.dropout = Dropout(p = self.dropout_rate)
        self.device_objects = {'self.bert': self.bert, 'self.dropout': self.dropout}

        self.initialize_parameters()


    def initialize_parameters(self):
        """
        Initialize weights to Xavier uniform, other to 1.
        """
        for k, v in self.device_objects.items(): 
            if k != 'self.bert':
                if hasattr(v, 'weight'):
                    if len(tuple(v.weight.data.shape)) > 1: 
                        v.weight.data = torch.nn.init.xavier_uniform_(v.weight.data)
                    else:
                        v.weight.data = torch.nn.init.xavier_uniform_(v.weight.data)
                if hasattr(v, 'bias'):
                    if not isinstance(v.bias, bool):
                        v.bias.data = torch.nn.init.zeros_(v.bias.data)


    def forward(self, batch):
        """
		Place holder for Base class
        @param batch: tuple or list, tuples of input_ids, attention_maksks, labels, and sentence legnth 
        """
        print("Method not implemented for base class!")


    def build(self, device, optimizer, loss_functions:list):
        """
        param loss_functions: list of loss function objects in torch in order of tasks
        """
        self.device = device
        self.opt = optimizer
        self.loss_functions = loss_functions
    
        # Move models to device
        for x in self.device_objects.values():
            x.to(device)


    def evaluate(self, X, batch_size=20, verbose=False):
        val_loss = [0] * self.num_tasks
        output_ls = []
        label_ls = []

        # Switch to eval mode
        for x in self.device_objects.values():
            x.eval()

        # Memory restriction, use data loader 
        val_dataloader = DataLoader(X, 
                                    # sampler=RandomSampler(X), 
                                    shuffle = False, 
                                    batch_size = batch_size)

        for index, batch in enumerate(val_dataloader):
            # No gradient needed
            with torch.no_grad():
                logits, labels = self.forward(batch)
                
                # Calculate losses
                for i in range(self.num_tasks):
                    batch_loss = self.loss_functions[i](logits[i], labels[i])
                    val_loss[i] += batch_loss.cpu().item()
                     
                label_ls.append([label.cpu().numpy().tolist() for label in labels ])
                output_ls.append([logit.cpu().numpy().tolist() for logit in logits])
                
        return val_loss, output_ls, label_ls


    def train(self, X, num_epochs, batch_size=20, scheduler=None, X_val = None, verbose=False): 
        """
        param X: tensor, dataset that contains all inputs, attention masks, and labels 
        """
        self.scheduler = scheduler
        # create Data loader 
        train_dataloader = DataLoader(X,
                         # sampler=RandomSampler(X),
                          shuffle=False,
                          batch_size=batch_size)
        
        loss_history = {'train': [], 'val': []}

        for e in range(num_epochs):
            print("Running Epoch ", e)
            # Switch to train mode
            [x.train() for x in self.device_objects.values()] 

            train_loss = [0] * self.num_tasks
            start =  time.time()

            for step, batch in enumerate(train_dataloader):
                self.bert.zero_grad()

                logits, labels = self.forward(batch)
                overall_loss = 0
                # Calculate losses
                for i in range(self.num_tasks):
                    batch_loss = self.loss_functions[i](logits[i], labels[i])
                    overall_loss += batch_loss
                    train_loss[i] += batch_loss.cpu().item()

                # backgrad to update parameters
                overall_loss.backward()
                [torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0) for layer in self.device_objects.values()] 
                self.opt.step()
                
                # weight decay 
                if self.scheduler is not None:
                    self.scheduler.step()

            # Results on valdiation set
            val_loss = []
            if X_val is not None:
                val_loss, _, _ = self.evaluate(X_val, batch_size)

            # Append to overal_losses
            for i in range(self.num_tasks):
                self.train_losses[i].append(train_loss[i])
                if X_val is not None: 
                    self.val_losses[i].append(val_loss[i])
                
            duration = time.time()- start
            if verbose: 
                print("Training loss : {}. Validation loss: {}. Duration: {}".format(train_loss,
                                                                                 val_loss,  
                                                                                 duration))
            loss_history['train'].append(train_loss)
            loss_history['val'].append(val_loss)

        return loss_history

    def save(self, path:str, init_yn=False): 
        """
        param init_yn: bool, True if weights are saved at initialization
        """
        moment = datetime.now()
        if init_yn: 
            filename = path + '/' + self.model_type + '_init.pth'
        else:
            filename = path + '/' + self.model_type + '_' + moment.strftime('%m_%d_%y_%H%M') + '.pth'
        # print(filename)
        params = {'args': dict(bert_type=self.bert_type, 
                               num_labels= self.num_labels,  
                               max_len = self.max_len,
                               attn_yn= self.attn_yn ,
                               dropout_rate=self.dropout_rate
                               )
                }
        layer_params = {k.replace('self.','') + '_state_dict' : v.state_dict() for k, v in self.device_objects.items()}
        params.update(layer_params)
        torch.save(params, filename)

        if init_yn: return filename


    @staticmethod
    def load(model_path: str):
        """
        Static method does not need self
        param_path: str, dirctory including extension of saved model files
        """
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        loaded_model = Bert_base(**args)
        # load state_dict in each layer
        for k, v in loaded_model.device_objects.items():
            v.load_state_dict(params[k.replace('self.', '') + '_state_dict'])

        return loaded_model


    def load_parameters(self, model_path:str):
        """
        Load parameters of the exact same models with the same tasks
        """
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        print("Parameters loaded for layers:", [k.replace('self.', '') for k in  params if k != 'args'] )
        # load state dict for each layer
        for k, v in self.device_objects.items():
            v.load_state_dict(params[k.replace('self.', '') + '_state_dict'])


    def load_states(self, sdict: dict):
        """
        Load state dicts from other models that have similar, but not
        necessarily same architecture and tasks 
        """
        for k, v in sdict.items(): 
            if k in self.device_objects:
                print("Loading state_dict for {}".format(k))
                self.device_objects[k].load_state_dict(v.state_dict())
                

############################## BERT-NN BASE ##############################
class BertNN(Bert_base):
    def __init__(self, bert_type, num_labels:list, max_len, attn_yn=False, dropout_rate = 0.2):
        super(BertNN, self).__init__(bert_type, num_labels, max_len, attn_yn, dropout_rate)
        self.model_type = 'Bert_NN'
        self.bert_type = bert_type
        if len(num_labels) == 1:
            print('Single-task training')
        else: 
            print('Multitask training: {} tasks '.format(len(num_labels)))

        self.num_labels = num_labels
        self.num_tasks = len(self.num_labels)
        self.train_losses = [[]] * self.num_tasks
        self.val_losses = [[]] * self.num_tasks

        self.attn_yn = attn_yn
        self.max_len = torch.tensor([max_len])
        self.dropout_rate = 0.2
        # BERT 
        self.bert = BertModel.from_pretrained(self.bert_type,
                                                   output_attentions = attn_yn, 
                                                   output_hidden_states=False
                                              )
        self.dropout = Dropout(p = self.dropout_rate)
        self.device_objects = {'self.bert': self.bert, 'self.dropout': self.dropout}
        # First Linear 
        self.linear_out_dim = 384
        self.fc = Linear(in_features=self.bert.config.hidden_size,
                         out_features=self.linear_out_dim,
                         bias=True)
        self.device_objects['self.fc'] = self.fc
        # Classifier layers
        self.fc_dict = {
                        'self.fc'+str(i) : Linear(in_features=self.linear_out_dim,
                                            out_features = n,
                                            bias=True)
                        for i,n in enumerate(self.num_labels)
                     }
                              
        self.device_objects.update(self.fc_dict)
        self.initialize_parameters()


    def forward(self, batch):
        """
        One pass forward using data in batch. 
        NOTE: Should ONLY BE USED inside evaluate or train function
        @param batch: tuple or list, tuples of input_ids, attention_maksks, labels, and sentence legnth 
        """
        sent_lens = batch[-1]
        batch = tuple(t.to(self.device) for t in batch[:-1])
        input_ids, attn_masks = batch[:2]
        labels = batch[2:]
        # print("sent_lens", sent_lens)

        # Bert
        bert_outputs = self.bert(input_ids, token_type_ids=None, 
                            attention_mask = attn_masks)
        encoded_layers = bert_outputs.pooler_output
        # Linear layer
        output_hidden = self.fc(encoded_layers)
        # Drop out
        output_hidden = self.dropout(output_hidden)
        # Classifier layer
        outputs = []
        for i in range(self.num_tasks):
            output_temp = self.fc_dict['self.fc'+str(i)](output_hidden)
            # If using Cross-Entropy loss, then do NOT normalize output
            outputs.append(output_temp)

        return outputs, labels


    @staticmethod
    def load(model_path: str):
        """
        Static method does not need self
        param_path: str, dirctory including extension of saved model files
        """
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        loaded_model = BertNN(**args)
        # load state_dict in each layer
        for k, v in loaded_model.device_objects.items():
            v.load_state_dict(params[k.replace('self.', '') + '_state_dict'])

        return loaded_model



############################## BERT CNN ##############################
class BertCNN(Bert_base):
    def __init__(self, bert_type, num_labels:list, max_len, attn_yn=False, dropout_rate = 0.2):
        super(BertCNN, self).__init__(bert_type, num_labels, max_len, attn_yn, dropout_rate)
        self.model_type = 'Bert_CNN'
        self.bert_type = bert_type
        if len(num_labels) == 1:
            print('Single-task training')
        else: 
            print('Multitask training: {} tasks '.format(len(num_labels)))

        self.num_labels = num_labels
        self.num_tasks = len(self.num_labels)
        self.train_losses = [[]] * self.num_tasks
        self.val_losses = [[]] * self.num_tasks

        self.attn_yn = attn_yn
        self.max_len = torch.tensor([max_len])
        self.dropout_rate = 0.2
        # BERT -> LSTM -> Dropout -> FC
        self.bert = BertModel.from_pretrained(self.bert_type,
                                                   output_attentions = attn_yn, 
                                                   output_hidden_states=True
                                              )
        self.dropout = Dropout(p = self.dropout_rate)
        self.device_objects = {'self.bert': self.bert, 'self.dropout': self.dropout}
        ## CNN 
        self.conv1 = Conv2d(12, 3, kernel_size=3, stride = 1, padding=1)   
        self.pool1 = MaxPool2d(kernel_size=3, stride=1, padding=0)     
        self.relu1 = ReLU()
        self.cnn_layers = {'self.conv1': self.conv1, 
                           'self.pool1' : self.pool1, 
                           'self.relu1': self.relu1
                           }
        
        self.device_objects.update(self.cnn_layers)
        # First linear 
        self.linear_input_dim = self.num_tasks * 223 * 766
        self.linear_out_dim   = 300
        self.fc = Linear(in_features=self.linear_input_dim, out_features=self.linear_out_dim)
        self.device_objects['self.fc'] = self.fc
        # Final linear layers
        self.fc_dict = {
                        'self.fc'+str(i) : Linear(in_features=self.linear_out_dim,
                                            out_features = n,
                                            bias=True)
                        for i,n in enumerate(self.num_labels)
                     }
        self.device_objects.update(self.fc_dict)
        # initialize all weights and bias
        self.initialize_parameters()


    def forward(self, batch):
        """
        One pass forward using data in batch. 
        NOTE: Should ONLY BE USED inside evaluate or train function
        @param batch: tuple or list, tuples of input_ids, attention_maksks, labels, and sentence legnth 
        """
        sent_lens = batch[-1]
        batch = tuple(t.to(self.device) for t in batch[:-1])
        input_ids, attn_masks = batch[:2]
        labels = batch[2:]
        # print("sent_lens", sent_lens)

        # Bert
        bert_outputs = self.bert(input_ids, token_type_ids=None, 
                            attention_mask = attn_masks)
        attn_hidden_states = bert_outputs[2][1:]
        # reshape tensor like images
        attn_hidden_states = torch.stack(attn_hidden_states)
        output_hidden = torch.transpose(attn_hidden_states, 0,1)
        # To CNN 
        for cnn_item in self.cnn_layers:
            output_hidden = self.cnn_layers[cnn_item](output_hidden)
        # Flatten
        output_hidden = output_hidden.view(-1, self.linear_input_dim)
        # To first linear layer
        output_hidden = self.fc(output_hidden)
        # Drop out
        output_hidden = self.dropout(output_hidden)
        # Classifier layer
        outputs = []
        for i in range(self.num_tasks):
            output_temp = self.fc_dict['self.fc'+str(i)](output_hidden)
            # If using Cross-Entropy loss, then do NOT normalize output
            outputs.append(output_temp)

        # return output_hidden
        return outputs, labels


    @staticmethod
    def load(model_path: str):
        """
        Static method does not need self
        param_path: str, dirctory including extension of saved model files
        """
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        # Change to corresponding class 
        loaded_model = BertCNN(**args)
        # load state_dict in each layer
        for k, v in loaded_model.device_objects.items():
            v.load_state_dict(params[k.replace('self.', '') + '_state_dict'])

        return loaded_model



############################# BERT LSTM ##############################
class BertLSTM(Bert_base):
    def __init__(self, bert_type, num_labels:list, max_len, attn_yn=False, dropout_rate = 0.2):
        super(BertLSTM, self).__init__(bert_type, num_labels, max_len, attn_yn, dropout_rate)
        self.model_type = 'Bert_LSTM'
        self.bert_type = bert_type
        if len(num_labels) == 1:
            print('Single-task training')
        else: 
            print('Multitask training: {} tasks '.format(len(num_labels)))

        self.num_labels = num_labels
        self.num_tasks = len(self.num_labels)
        self.train_losses = [[]] * self.num_tasks
        self.val_losses = [[]] * self.num_tasks

        self.attn_yn = attn_yn
        self.max_len = torch.tensor([max_len])
        self.dropout_rate = 0.2
        # BERT 
        self.bert = BertModel.from_pretrained(self.bert_type,
                                                   output_attentions = attn_yn, 
                                                   output_hidden_states=True
                                              )
        self.dropout = Dropout(p = self.dropout_rate)
        self.device_objects = {'self.bert': self.bert, 'self.dropout': self.dropout}
        ## LSTM
        self.lstm_input_size = self.bert.config.hidden_size
        self.lstm_hidden_size = self.bert.config.hidden_size
        self.lstm = LSTM(input_size = self.lstm_input_size,
                        hidden_size = self.lstm_hidden_size, 
                         bidirectional = True )
        self.device_objects['self.lstm'] = self.lstm
        # Final linear layers
        self.fc_dict = {
                        'self.fc'+str(i) : Linear(in_features=2*self.lstm_hidden_size,
                                            out_features = n,
                                            bias=True)
                        for i,n in enumerate(self.num_labels)
                     }
        self.device_objects.update(self.fc_dict)
        # initialize all weights and bias
        self.initialize_parameters()


    def forward(self, batch):
        """
        One pass forward using data in batch. 
        NOTE: Should ONLY BE USED inside evaluate or train function
        @param batch: tuple or list, tuples of input_ids, attention_maksks, labels, and sentence legnth 
        """
        sent_lens = batch[-1]
        batch = tuple(t.to(self.device) for t in batch[:-1])
        input_ids, attn_masks = batch[:2]
        labels = batch[2:]
        # print("sent_lens", sent_lens)

        # Bert
        bert_outputs = self.bert(input_ids, token_type_ids=None, 
                            attention_mask = attn_masks)
        
        # LSTM 
        encoded_layers = bert_outputs.last_hidden_state
        encoded_layers = encoded_layers.permute(1,0,2)
        enc_hidden, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(encoded_layers,
                                                                                      sent_lens,
                                                                                      enforce_sorted=False))
        # Flatten 
        output_hidden = torch.cat((last_hidden[0,:,:], last_hidden[1,:,:]), dim=1)
        # Drop out
        output_hidden = self.dropout(output_hidden)
        # Classifier layer
        outputs = []
        for i in range(self.num_tasks):
            output_temp = self.fc_dict['self.fc'+str(i)](output_hidden)
            # If using Cross-Entropy loss, then do NOT normalize output
            outputs.append(output_temp)

        # return output_hidden
        return outputs, labels


    @staticmethod
    def load(model_path: str):
        """
        Static method does not need self
        param_path: str, dirctory including extension of saved model files
        """
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        # Change to corresponding class 
        loaded_model = BertLSTM(**args)
        # load state_dict in each layer
        for k, v in loaded_model.device_objects.items():
            v.load_state_dict(params[k.replace('self.', '') + '_state_dict'])

        return loaded_model


############################# ROBERTA BASE ##############################
class Roberta_base(Bert_base):
    def __init__(self, bert_type, num_labels:list, max_len, attn_yn=False, dropout_rate = 0.2):
        super(Bert_base, self).__init__()
        self.model_type = 'Robert_base'
        self.bert_type = bert_type
        if len(num_labels) == 1:
            print('Sigle-task training')
        else: 
            print('Multitask training: {} tasks '.format(len(num_labels)))

        self.num_labels = num_labels
        self.num_tasks = len(self.num_labels)
        self.train_losses = [[]] * self.num_tasks
        self.val_losses = [[]] * self.num_tasks

        self.attn_yn = attn_yn
        self.max_len = torch.tensor([max_len])
        self.dropout_rate = 0.2
        # BERT 
        self.bert = RobertaModel.from_pretrained(self.bert_type,
                                                #    output_attentions = attn_yn, 
                                                #    output_hidden_states=False
                                              )
        self.dropout = Dropout(p = self.dropout_rate)
        self.device_objects = {'self.bert': self.bert, 'self.dropout': self.dropout}

        self.initialize_parameters()

############################# ROBERTA NN ##############################
class RobertaNN(Roberta_base):
    def __init__(self, bert_type, num_labels:list, max_len, attn_yn=False, dropout_rate = 0.2):
        super(RobertaNN, self).__init__(bert_type, num_labels, max_len, attn_yn, dropout_rate)
        self.model_type = 'Roberta_NN'
        self.bert_type = bert_type
        if len(num_labels) == 1:
            print('Single-task training')
        else: 
            print('Multitask training: {} tasks '.format(len(num_labels)))

        self.num_labels = num_labels
        self.num_tasks = len(self.num_labels)
        self.train_losses = [[]] * self.num_tasks
        self.val_losses = [[]] * self.num_tasks

        self.attn_yn = attn_yn
        self.max_len = torch.tensor([max_len])
        self.dropout_rate = dropout_rate
        # BERT 
        self.bert = RobertaModel.from_pretrained(self.bert_type,
                                                #    output_attentions = attn_yn, 
                                                #    output_hidden_states=False
                                              )
        self.dropout = Dropout(p = self.dropout_rate)
        self.device_objects = {'self.bert': self.bert, 'self.dropout': self.dropout}
        # First Linear 
        self.linear_out_dim = 384
        self.fc = Linear(in_features=self.bert.config.hidden_size,
                         out_features=self.linear_out_dim,
                         bias=True)
        self.device_objects['self.fc'] = self.fc
        # Classifier layers
        self.fc_dict = {
                        'self.fc'+str(i) : Linear(in_features=self.linear_out_dim,
                                            out_features = n,
                                            bias=True)
                        for i,n in enumerate(self.num_labels)
                     }
                              
        self.device_objects.update(self.fc_dict)
        self.initialize_parameters()


    def forward(self, batch):
        """
        One pass forward using data in batch. 
        NOTE: Should ONLY BE USED inside evaluate or train function
        @param batch: tuple or list, tuples of input_ids, attention_maksks, labels, and sentence legnth 
        """
        sent_lens = batch[-1]
        batch = tuple(t.to(self.device) for t in batch[:-1])
        input_ids, attn_masks = batch[:2]
        labels = batch[2:]
        # print("sent_lens", sent_lens)

        # Bert
        bert_outputs = self.bert(input_ids, token_type_ids=None, 
                            attention_mask = attn_masks)
        encoded_layers = bert_outputs.pooler_output
        # Linear layer
        output_hidden = leaky_relu(self.fc(encoded_layers))
        # Drop out
        output_hidden = self.dropout(output_hidden)
        # Classifier layer
        outputs = []
        for i in range(self.num_tasks):
            output_temp = self.fc_dict['self.fc'+str(i)](output_hidden)
            # If using Cross-Entropy loss, then do NOT normalize output
            outputs.append(output_temp)

        return outputs, labels


    @staticmethod
    def load(model_path: str):
        """
        Static method does not need self
        param_path: str, dirctory including extension of saved model files
        """
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        loaded_model = RobertaNN(**args)
        # load state_dict in each layer
        for k, v in loaded_model.device_objects.items():
            v.load_state_dict(params[k.replace('self.', '') + '_state_dict'])

        return loaded_model


############################# ROBERTA LSTM ##############################
class RobertaLSTM(Roberta_base):
    def __init__(self, bert_type, num_labels:list, max_len, attn_yn=False, dropout_rate = 0.2):
        super(RobertaLSTM, self).__init__(bert_type, num_labels, max_len, attn_yn, dropout_rate)
        self.model_type = 'Roberta_LSTM'
        self.bert_type = bert_type
        if len(num_labels) == 1:
            print('Single-task training')
        else: 
            print('Multitask training: {} tasks '.format(len(num_labels)))

        self.num_labels = num_labels
        self.num_tasks = len(self.num_labels)
        self.train_losses = [[]] * self.num_tasks
        self.val_losses = [[]] * self.num_tasks

        self.attn_yn = attn_yn
        self.max_len = torch.tensor([max_len])
        self.dropout_rate = 0.2
        # BERT 
        self.bert = RobertaModel.from_pretrained(self.bert_type,
                                                #    output_attentions = attn_yn, 
                                                #    output_hidden_states=True
                                              )
        self.dropout = Dropout(p = self.dropout_rate)
        self.device_objects = {'self.bert': self.bert, 'self.dropout': self.dropout}
        ## LSTM
        self.lstm_input_size = self.bert.config.hidden_size
        self.lstm_hidden_size = self.bert.config.hidden_size
        self.lstm = LSTM(input_size = self.lstm_input_size,
                        hidden_size = self.lstm_hidden_size, 
                         bidirectional = True )
        self.device_objects['self.lstm'] = self.lstm
        # Classifier layers
        self.fc_dict = {
                        'self.fc'+str(i) : Linear(in_features=2*self.lstm_hidden_size,
                                            out_features = n,
                                            bias=True)
                        for i,n in enumerate(self.num_labels)
                     }
        self.device_objects.update(self.fc_dict)
        # initialize all weights and bias
        self.initialize_parameters()


    def forward(self, batch):
        """
        One pass forward using data in batch. 
        NOTE: Should ONLY BE USED inside evaluate or train function
        @param batch: tuple or list, tuples of input_ids, attention_maksks, labels, and sentence legnth 
        """
        sent_lens = batch[-1]
        batch = tuple(t.to(self.device) for t in batch[:-1])
        input_ids, attn_masks = batch[:2]
        labels = batch[2:]
        # print("sent_lens", sent_lens)

        # Bert
        bert_outputs = self.bert(input_ids, token_type_ids=None, 
                            attention_mask = attn_masks)
        
        # LSTM 
        encoded_layers = bert_outputs.last_hidden_state
        encoded_layers = encoded_layers.permute(1,0,2)
        enc_hidden, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(encoded_layers,
                                                                                      sent_lens,
                                                                                      enforce_sorted=False))
        # Flatten 
        output_hidden = torch.cat((last_hidden[0,:,:], last_hidden[1,:,:]), dim=1)
        # Drop out
        output_hidden = self.dropout(output_hidden)
        # Classifier layer
        outputs = []
        for i in range(self.num_tasks):
            output_temp = self.fc_dict['self.fc'+str(i)](output_hidden)
            # If using Cross-Entropy loss, then do NOT normalize output
            outputs.append(output_temp)

        # return output_hidden
        return outputs, labels


    @staticmethod
    def load(model_path: str):
        """
        Static method does not need self
        param_path: str, dirctory including extension of saved model files
        """
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        loaded_model = RobertaLSTM(**args)
        # load state_dict in each layer
        for k, v in loaded_model.device_objects.items():
            v.load_state_dict(params[k.replace('self.', '') + '_state_dict'])

        return loaded_model

############################# ROBERTA CNN ##############################
class RobertaCNN(Roberta_base):
    def __init__(self, bert_type, num_labels:list, max_len, attn_yn=False, dropout_rate = 0.2):
        super(RobertaCNN, self).__init__(bert_type, num_labels, max_len, attn_yn, dropout_rate)
        self.model_type = 'Roberta_CNN'
        self.bert_type = bert_type
        if len(num_labels) == 1:
            print('Single-task training')
        else: 
            print('Multitask training: {} tasks '.format(len(num_labels)))

        self.num_labels = num_labels
        self.num_tasks = len(self.num_labels)
        self.train_losses = [[]] * self.num_tasks
        self.val_losses = [[]] * self.num_tasks

        self.attn_yn = attn_yn
        self.max_len = torch.tensor([max_len])
        self.dropout_rate = 0.2
        # BERT 
        self.bert = RobertaModel.from_pretrained(self.bert_type,
                                                #    output_attentions = attn_yn, 
                                                   output_hidden_states=True
                                              )
        self.dropout = Dropout(p = self.dropout_rate)
        # self.relu    = ReLU()
        self.device_objects = {'self.bert': self.bert, 'self.dropout': self.dropout
        }
        ## CNN 
        self.conv1 = Conv2d(6, 3, kernel_size=3, stride = 1, padding=1)   
        self.pool1 = MaxPool2d(kernel_size=3, stride=1, padding=0)     
        self.relu1 = LeakyReLU()
        self.cnn_layers = {'self.conv1': self.conv1, 
                           'self.pool1' : self.pool1, 
                           'self.relu1': self.relu1
                           }
        self.device_objects.update(self.cnn_layers)
        # First linear 
        self.linear_input_dim = 3 * 223 * 766
        self.linear_out_dim   = 384
        self.fc = Linear(in_features=self.linear_input_dim, out_features=self.linear_out_dim)
        self.device_objects['self.fc'] = self.fc
        # Final linear layers
        self.fc_dict = {
                        'self.fc'+str(i) : Linear(in_features=self.linear_out_dim,
                                            out_features = n,
                                            bias=True)
                        for i,n in enumerate(self.num_labels)
                     }
        self.device_objects.update(self.fc_dict)
        # initialize all weights and bias
        self.initialize_parameters()


    def forward(self, batch):
        """
        One pass forward using data in batch. 
        NOTE: Should ONLY BE USED inside evaluate or train function
        @param batch: tuple or list, tuples of input_ids, attention_maksks, labels, and sentence legnth 
        """
        sent_lens = batch[-1]
        batch = tuple(t.to(self.device) for t in batch[:-1])
        input_ids, attn_masks = batch[:2]
        labels = batch[2:]
        # print("sent_lens", sent_lens)

        # Bert
        bert_outputs = self.bert(input_ids, token_type_ids=None, 
                            attention_mask = attn_masks)
        attn_hidden_states = bert_outputs[2][-6:]
        # reshape tensor like images
        attn_hidden_states = torch.stack(attn_hidden_states)
        output_hidden = torch.transpose(attn_hidden_states, 0,1)
        # To CNN 
        for cnn_item in self.cnn_layers:
            output_hidden = self.cnn_layers[cnn_item](output_hidden)
        # Flatten
        output_hidden = output_hidden.view(-1, self.linear_input_dim)
        # To first linear layer
        output_hidden = leaky_relu(self.fc(output_hidden))
        # Drop out
        output_hidden = self.dropout(output_hidden)
        # Classifier layer
        outputs = []
        for i in range(self.num_tasks) :
            output_temp = self.fc_dict['self.fc'+str(i)](output_hidden)
            # If using Cross-Entropy loss, then do NOT normalize output
            outputs.append(output_temp)

        # return output_hidden
        return outputs, labels

    @staticmethod
    def load(model_path: str):
        """
        Static method does not need self
        param_path: str, dirctory including extension of saved model files
        """
        params = torch.load(model_path, map_location = lambda storage, loc: storage)
        args = params['args']
        loaded_model = RobertaCNN(**args)
        # load state_dict in each layer
        for k, v in loaded_model.device_objects.items():
            v.load_state_dict(params[k.replace('self.', '') + '_state_dict'])

        return loaded_model
