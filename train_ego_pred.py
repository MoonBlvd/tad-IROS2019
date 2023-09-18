import sys
import os 
import numpy as np
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchsummaryX import summary

from lib.utils.train_val_utils import train_ego_pred, val_ego_pred
from lib.models.rnn_ed import EgoRNNED
from lib.utils.fol_dataloader import HEVIEgoDataset
from config.config import * 

from tensorboardX import SummaryWriter


print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load args
args = parse_args()

# initialize model
model = EgoRNNED(args).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

# initialize datasets
dataloader_params ={
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers
    }
# train_set = HEVIDataset(args, 'train')
# train_gen = data.DataLoader(train_set, **dataloader_params)
# print("Number of training samples:", train_set.__len__())

val_set = HEVIEgoDataset(args, 'val')
val_gen = data.DataLoader(val_set, **dataloader_params)
print("Number of validation samples:", val_set.__len__())

# print model summary
summary(model, torch.zeros(1, args.segment_len, 3).to(device))

# summary writer
writer = SummaryWriter('summary/ego_pred/exp-1')

# train
all_val_loss = []
min_loss = 1e6
best_model = None
for epoch in range(1, args.nb_ego_pred_epoch+1):
    # regenerate the training dataset 
    train_set = HEVIEgoDataset(args, 'train')
    train_gen = data.DataLoader(train_set, **dataloader_params)
    print("Number of training samples:", train_set.__len__())

    start = time.time()
    # train
    train_loss = train_ego_pred(epoch, model, optimizer, train_gen,)
    writer.add_scalar('data/train_loss', train_loss, epoch)
    # print('====> Epoch: {} object pred loss: {:.4f}'.format(epoch, train_loss))
    # val
    val_loss = val_ego_pred(epoch, model, val_gen)
    writer.add_scalar('data/val_loss', val_loss, epoch)
#     print('====> Epoch: {} validation loss: {:.4f}'.format(epoch, val_loss))
    all_val_loss.append(val_loss)

    # print time
    elipse = time.time() - start
    print("Elipse: ", elipse)

    # save checkpoints if loss decreases
    if val_loss < min_loss:
        try:
            os.remove(best_model)
        except:
            pass
        min_loss = val_loss
        saved_model_name = 'epoch_' + str(format(epoch,'03')) + \
                        '_loss_%.4f'%val_loss + '.pt'
        print("Saving checkpoints: " + saved_model_name)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, saved_model_name))

        best_model = os.path.join(args.checkpoint_dir, saved_model_name)
