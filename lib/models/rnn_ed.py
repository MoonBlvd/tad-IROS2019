import sys
import numpy as np
import copy

import torch
from torch import nn, optim
from torch.nn import functional as F
# from torch.autograd import Variable

import pickle as pkl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class EncoderGRU(nn.Module):
    def __init__(self, args):
        super(EncoderGRU, self).__init__()
        self.args = args
        self.enc = nn.GRUCell(input_size=self.args.input_embed_size,
                    hidden_size=self.args.enc_hidden_size)

    def forward(self, embedded_input, h_init):
        '''
        The encoding process
        Params:
            x: input feature, (batch_size, time, feature dims)
            h_init: initial hidden state, (batch_size, enc_hidden_size)
        Returns:
            h: updated hidden state of the next time step, (batch.size, enc_hiddden_size)
        '''
        h = self.enc(embedded_input, h_init)
        return h
        
class DecoderGRU(nn.Module):
    def __init__(self, args):
        super(DecoderGRU, self).__init__()
        self.args = args
        # PREDICTOR INPUT FC
        self.hidden_to_pred_input = nn.Sequential(nn.Linear(self.args.dec_hidden_size,
                                                            self.args.predictor_input_size),
                                                  nn.ReLU())
        
        # PREDICTOR DECODER
        self.dec = nn.GRUCell(input_size=self.args.predictor_input_size,
                                        hidden_size=self.args.dec_hidden_size)
        
        # PREDICTOR OUTPUT
        if self.args.non_linear_output:
            self.hidden_to_pred = nn.Sequential(nn.Linear(self.args.dec_hidden_size, 
                                                            self.args.pred_dim),
                                                nn.Tanh())
        else:
            self.hidden_to_pred = nn.Linear(self.args.dec_hidden_size, 
                                                            self.args.pred_dim)
                
    def forward(self, h, embedded_ego_pred=None):
        '''
        A RNN preditive model for future observation prediction
        Params:
            h: hidden state tensor from the encoder, (batch_size, enc_hidden_size)
            embedded_ego_pred: (batch_size, pred_timesteps, input_embed_size)
        '''
        output = torch.zeros(h.shape[0], self.args.pred_timesteps, self.args.pred_dim).to(device)

        all_pred_h = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.dec_hidden_size]).to(device)
        all_pred_inputs = torch.zeros([h.shape[0], self.args.pred_timesteps, self.args.predictor_input_size]).to(device)
        
        # initial predict input is zero???
        pred_inputs = torch.zeros(h.shape[0], self.args.predictor_input_size).to(device) #self.hidden_to_pred_input(h)
        for i in range(self.args.pred_timesteps):
            if self.args.with_ego:
                pred_inputs = (embedded_ego_pred[:, i, :] + pred_inputs)/2 # average concat of future ego motion and prediction inputs
            all_pred_inputs[:, i, :] = pred_inputs
            h = self.dec(pred_inputs, h)
            
            pred_inputs = self.hidden_to_pred_input(h)

            all_pred_h[:,i,:] = h

            output[:,i,:] = self.hidden_to_pred(h)

        return output, all_pred_h, all_pred_inputs

class FolRNNED(nn.Module):
    '''Future object localization module'''
    def __init__(self, args):
        super(FolRNNED, self).__init__()

        # get args and process
        self.args = copy.deepcopy(args)
        self.box_enc_args = copy.deepcopy(args)
        self.flow_enc_args = copy.deepcopy(args)

        if self.args.enc_concat_type == 'cat':
            self.args.dec_hidden_size = self.args.box_enc_size + self.args.flow_enc_size
        else:
            if self.args.box_enc_size != self.args.flow_enc_size:
                raise ValueError('Box encoder size %d != flow encoder size %d'
                                    %(self.args.box_enc_size,self.args.flow_enc_size))
            else:
                self.args.dec_hidden_size = self.args.box_enc_size
        
        self.box_enc_args.enc_hidden_size = self.args.box_enc_size
        self.flow_enc_args.enc_hidden_size = self.args.flow_enc_size
        
        # initialize modules
        self.box_encoder = EncoderGRU(self.box_enc_args)
        self.flow_encoder = EncoderGRU(self.flow_enc_args)
        self.args.non_linear_output = True
        self.predictor = DecoderGRU(self.args)

        # initialize other layers
        # self.leaky_relu = nn.LeakyReLU(0.1)
        self.box_embed = nn.Sequential(nn.Linear(4, self.args.input_embed_size), # size of box input is 4
                                        nn.ReLU()) # nn.LeakyReLU(0.1)
        self.flow_embed = nn.Sequential(nn.Linear(50, self.args.input_embed_size), # size of flow input is 50=5*5*2
                                        nn.ReLU()) # nn.LeakyReLU(0.1)
        self.ego_pred_embed = nn.Sequential(nn.Linear(3, self.args.input_embed_size), # size of ego input is 3
                                        nn.ReLU()) # nn.LeakyReLU(0.1)

        # if args.with_ego:
        #     # initialize ego motion predictor and load the pretrained model
        #     print("Initializing pre-trained ego motion predictor!")
        #     self.ego_predictor = EgoRNNED(self.args)
        #     self.ego_predictor.load_state_dict(torch.load(self.args.best_ego_pred_model))
        #     print("Pre-trained ego_motion predictor done!")

    def forward(self, box, flow, ego_pred):#ego_motion):
        '''
        The RNN encoder decoder model rewritten from fvl2019icra-keras
        Params:
            box: (batch_size, segment_len, 4)
            flow: (batch_size, segment_len, 5, 5, 2)
            ego_pred: (batch_size, segment_len, pred_timesteps, 3) or None
            
            for training and validation, segment_len is large, e.g. 10
            for online testing, segment_len=1
        return:
            fol_predictions: predicted with shape (batch_size, segment_len, pred_timesteps, pred_dim)
        '''
        self.args.batch_size = box.shape[0]
        if len(flow.shape) > 3:
            flow = flow.view(self.args.batch_size, self.args.segment_len, -1)
        embedded_box_input= self.box_embed(box)
        embedded_flow_input= self.flow_embed(flow)

        embedded_ego_input = self.ego_pred_embed(ego_pred) # (batch_size, segment_len, pred_timesteps, input_embed_size) 

        # initialize hidden states as zeros
        box_h = torch.zeros(self.args.batch_size, self.args.box_enc_size).to(device)
        flow_h = torch.zeros(self.args.batch_size, self.args.flow_enc_size).to(device)
        
        # a zero tensor used to save fol prediction
        fol_predictions = torch.zeros(self.args.batch_size, 
                                    self.args.segment_len, 
                                    self.args.pred_timesteps, 
                                    self.args.pred_dim).to(device)
        
        # run model iteratively, predict T future frames at each time 
        for i in range(self.args.segment_len):
            # Box and Flow Encode
            box_h = self.box_encoder(embedded_box_input[:,i,:], box_h)
            flow_h = self.flow_encoder(embedded_flow_input[:,i,:], flow_h)

            # Concat
            if self.args.enc_concat_type == 'cat':
                hidden_state = torch.cat((box_h, flow_h), dims=1)
            elif self.args.enc_concat_type in ['sum', 'avg', 'average']:
                hidden_state = (box_h + flow_h) / 2
            else:
                raise NameError(self.args.enc_concat_type, ' is unknown!!')
            
            # Decode
            if self.args.with_ego:
                output, _, _ = self.predictor(hidden_state, embedded_ego_input[:,i,:,:])
            else:
                # predict without future ego motion
                output, _, _ = self.predictor(hidden_state, None)
            
            fol_predictions[:,i,:,:] = output
        return fol_predictions
    
    def predict(self, box, flow, box_h, flow_h, ego_pred):
        '''
        predictor function, run forward inference to predict the future bboxes
        Params:
            box: (1, 4)
            flow: (1, 1, 5, 5, 2)
            ego_pred: (1, pred_timesteps, 3)
        return:
            box_changes:()
            box_h, 
            flow_h
        '''
        # self.args.batch_size = box.shape[0]
        if len(flow.shape) > 3:
            flow = flow.view(1, -1)
        embedded_box_input= self.box_embed(box)
        embedded_flow_input= self.flow_embed(flow)
        embedded_ego_input = None
        if self.args.with_ego:
            embedded_ego_input = self.ego_pred_embed(ego_pred)
        
        # run model iteratively, predict 5 future frames at each time 
        box_h = self.box_encoder(embedded_box_input, box_h)
        flow_h = self.flow_encoder(embedded_flow_input, flow_h)

        if self.args.enc_concat_type == 'cat':
            hidden_state = torch.cat((box_h, flow_h), dims=1)
        elif self.args.enc_concat_type in ['sum', 'avg', 'average']:
            hidden_state = (box_h + flow_h) / 2
        else:
            raise NameError(self.args.enc_concat_type, ' is unknown!!')
        
        box_changes, _, _ = self.predictor(hidden_state, embedded_ego_input)
        
         
        return box_changes, box_h, flow_h

class EgoRNNED(nn.Module):
    def __init__(self, args):
        super(EgoRNNED, self).__init__()

        self.args = copy.deepcopy(args)

        # update arguments for model 
        self.args.input_embed_size = self.args.ego_embed_size
        self.args.enc_hidden_size = self.args.ego_enc_size
        self.args.dec_hidden_size = self.args.ego_dec_size
        self.args.pred_dim = self.args.ego_dim
        self.args.predictor_input_size = self.args.ego_pred_input_size
        self.args.with_ego = False

        # initialize modules
        self.ego_encoder = EncoderGRU(self.args)

        # initialize other layers
        self.ego_embed = nn.Sequential(nn.Linear(3, self.args.ego_embed_size), # size of box input is 4
                                        nn.ReLU())#nn.LeakyReLU(0.1)

        # do not use activation when predicting future ego motion                                
        self.args.non_linear_output = False
        self.predictor = DecoderGRU(self.args)

    def forward(self, ego_x, image=None):
        '''
        The RNN encoder decoder model for ego motion prediction
        Params:
            ego_x: (batch_size, segment_len, ego_dim)
            image: (batch_size, segment_len, feature_dim) e.g. feature_dim = 1024
            
            for training and validation, segment_len is large, e.g. 10
            for online testing, segment_len=1
        return:
            predictions: predicted ego motion with shape (batch_size, segment_len, pred_timesteps, ego_dim)
        '''
        self.args.batch_size = ego_x.shape[0]
        
        # embedding
        embedded_ego_input= self.ego_embed(ego_x)
        
        # initialize hidden states as zeros
        ego_h = torch.zeros(self.args.batch_size, self.args.enc_hidden_size).to(device)
        
        # a zero tensor used to save outputs
        predictions = torch.zeros(self.args.batch_size, 
                                    self.args.segment_len, 
                                    self.args.pred_timesteps, 
                                    self.args.pred_dim).to(device)
        
        # run model iteratively, predict 5 future frames at each time 
        for i in range(self.args.segment_len):
            ego_h = self.ego_encoder(embedded_ego_input[:,i,:], ego_h)
            output, _, _ = self.predictor(ego_h)
            predictions[:,i,:,:] = output
            # break
        return predictions
    
    def predict(self, ego_x, ego_h, image=None):
        '''
        Params:
            ego_x: (1, 3)
            ego_h: (1, 64)
            #image: (1, 1, 1024) e.g. feature_dim = 1024
        returns:
            ego_changes: (pred_timesteps, 3)
            ego_h: (1, ego_enc_size)
        '''
        # embedding
        embedded_ego_input= self.ego_embed(ego_x)

        ego_h = self.ego_encoder(embedded_ego_input, ego_h)
        ego_changes, _, _ = self.predictor(ego_h)
        # break
        return ego_changes, ego_h

