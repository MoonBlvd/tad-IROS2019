import torch
from torch.nn import functional as F
from tqdm import tqdm
n_iters = 0

def train_fol(epoch, model, optimizer, train_gen, verbose=True):
    '''
    Validate future vehicle localization module 
    Params:
        epoch: int scalar
        model: The fol model as nn.Module
        optimizer:
        train_gen: validation data generator
    Returns:
        avg_obj_pred_loss: float scalar
    '''
    model.train() # Sets the module in training mode.
    total_train_loss = 0
    # n_iters = epoch * len(train_gen)
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(train_gen):
            input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = data
            # run forward
            predictions = model(input_bbox, input_flow)
            # compute loss
            object_pred_loss = rmse_loss_fol(predictions, target_bbox)
            total_train_loss += object_pred_loss.item()
            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            object_pred_loss.backward()
            optimizer.step()

            #write summery for tensorboardX
            if verbose and batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_gen.dataset),
                    100. * batch_idx / len(train_gen), object_pred_loss.item()/input_bbox.shape[0]))
    avg_obj_pred_loss = total_train_loss/len(train_gen.dataset)
    return avg_obj_pred_loss

def val_fol(epoch, model, val_gen, verbose=True):
    '''
    Validate future vehicle localization module 
    Params:
        epoch: int scalar
        model: The fol model as nn.Module
        train_gen: validation data generator
    Returns:
        avg_val_loss: float scalar
    '''
    model.eval() # Sets the module in training mode.
    total_val_loss = 0
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = data

            # run forward
            predictions = model(input_bbox, input_flow)
            # compute loss
            object_pred_loss = rmse_loss_fol(predictions, target_bbox)
            total_val_loss += object_pred_loss.item()
    
    avg_val_loss = total_val_loss/len(val_gen.dataset)

    if verbose:
        print('\nVal set: Average loss: {:.4f},\n'.format(avg_val_loss))

    return avg_val_loss

def train_ego_pred(epoch, model, optimizer, train_gen, verbose=True):
    '''
    Validate ego motion prediction module only
    Params:
        epoch: int scalar
        model: The ego motion prediction model as nn.Module
        optimizer:
        train_gen: validation data generator
    Returns:
        avg_ego_pred_loss: float scalar
    '''
    model.train() # Sets the module in training mode.
    total_train_loss = 0
    # n_iters = epoch * len(train_gen)
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(train_gen):
            input_ego_motion, target_ego_motion = data
            # run forward
            predictions = model(input_ego_motion)
            # compute loss
            ## TODO: should we use a weighted loss???
            ego_pred_loss = rmse_loss_fol(predictions, target_ego_motion)
            total_train_loss += ego_pred_loss.item()
            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            ego_pred_loss.backward()
            optimizer.step()

            #write summery for tensorboardX
            # writer.add_scalar('data/train_loss', object_pred_loss, n_iters)
            # n_iters += 1
            if verbose and batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_gen.dataset),
                    100. * batch_idx / len(train_gen), ego_pred_loss.item()/input_ego_motion.shape[0]))
    avg_ego_pred_loss = total_train_loss/len(train_gen.dataset)
    return avg_ego_pred_loss

def val_ego_pred(epoch, model, val_gen, verbose=True):
    '''
    Validate ego motion prediction module only
    Params:
        epoch: int scalar
        model: The ego motion prediction model as nn.Module
        val_gen: validation data generator
    Returns:
        avg_val_loss: float scalar
    '''
    model.eval() # Sets the module in training mode.
    total_val_loss = 0
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            input_ego_motion, target_ego_motion = data

            # run forward
            predictions = model(input_ego_motion)
            # compute loss
            ego_pred_loss = rmse_loss_fol(predictions, target_ego_motion)
            total_val_loss += ego_pred_loss.item()
    
    avg_val_loss = total_val_loss/len(val_gen.dataset)

    if verbose:
        print('\nVal set: Average loss: {:.4f},\n'.format(avg_val_loss))

    return avg_val_loss


def train_fol_ego(epoch, args, fol_model, ego_pred_model, optimizer, train_gen, verbose=True):
    '''
    Validate future vehicle localization module 
    Params:
        epoch: int scalar
        model: The fol model as nn.Module
        optimizer:
        train_gen: validation data generator
    Returns:
        avg_obj_pred_loss: float scalar
    '''
    fol_model.train() # Sets the module in training mode.
    ego_pred_model.train()

    avg_fol_loss = 0
    avg_ego_pred_loss = 0
    # n_iters = epoch * len(train_gen)
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(train_gen):
            input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = data
            
            # run forward
            ego_predictions = ego_pred_model(input_ego_motion) # the ego_predictions is actually ego changes 
                                                               # rather than the absolute ego motion values

            fol_predictions = fol_model(input_bbox, input_flow, ego_predictions)

            # compute loss
            fol_loss = rmse_loss_fol(fol_predictions, target_bbox)
            ego_pred_loss = rmse_loss_fol(ego_predictions, target_ego_motion)
            loss_to_optimize = args.lambda_fol * fol_loss +  args.lambda_ego * ego_pred_loss

            avg_fol_loss += fol_loss.item() 
            avg_ego_pred_loss += ego_pred_loss.item()

            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            loss_to_optimize.backward()
            optimizer.step()

            #write summery for tensorboardX
            # writer.add_scalar('data/train_loss', fol_loss, n_iters)
            # n_iters += 1
            if verbose and batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t FOL loss: {:.4f}\t Ego pred loss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_gen.dataset),
                    100. * batch_idx / len(train_gen), fol_loss.item()/input_bbox.shape[0], ego_pred_loss.item()/input_ego_motion.shape[0]))
        avg_fol_loss /= len(train_gen.dataset)
        avg_ego_pred_loss /= len(train_gen.dataset)
        avg_train_loss = avg_fol_loss + avg_ego_pred_loss
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t FOL loss: {:.4f}\t Ego pred loss: {:.4f} Total: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_gen.dataset),
                    100. * batch_idx / len(train_gen), avg_fol_loss, avg_ego_pred_loss, avg_train_loss))
        
    
    return avg_train_loss, avg_fol_loss, avg_ego_pred_loss

def val_fol_ego(epoch, args, fol_model, ego_pred_model, val_gen, verbose=True):
    '''
    Validate future vehicle localization module 
    Params:
        epoch: int scalar
        model: The fol model as nn.Module
        train_gen: validation data generator
    Returns:
        fol_loss: float scalar
        ego_pred_loss: float scalar
    '''
    fol_model.eval() # Sets the module in training mode.
    ego_pred_model.eval()

    fol_loss = 0
    ego_pred_loss = 0
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = data

            # run forward
            ego_predictions = ego_pred_model(input_ego_motion)
            fol_predictions = fol_model(input_bbox, input_flow, ego_predictions)

            # compute loss
            fol_loss += rmse_loss_fol(fol_predictions, target_bbox).item()
            ego_pred_loss += rmse_loss_fol(ego_predictions, target_ego_motion).item()

    fol_loss /= len(val_gen.dataset)
    ego_pred_loss /= len(val_gen.dataset)
    avg_val_loss = fol_loss + ego_pred_loss
    if verbose:
        print('\nVal set: Average FOL loss: {:.4f}, Average Ego pred loss: {:.4f}, Total: {:.4f}\n'.format(fol_loss, ego_pred_loss, avg_val_loss))

    return avg_val_loss, fol_loss, ego_pred_loss



def rmse_loss_fol(x_pred, x_true):
    '''
    Params:
        x_pred: (batch_size, segment_len, pred_timesteps, pred_dim)
        x_true: (batch_size, segment_len, pred_timesteps, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''

    L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=3))
    # sum over prediction time steps
    L2_all_pred = torch.sum(L2_diff, dim=2)

    # mean of each frames predictions
    L2_mean_pred = torch.mean(L2_all_pred, dim=1)

    # sum of all batches
    L2_mean_pred = torch.mean(L2_mean_pred, dim=0)

    return L2_mean_pred

    # return torch.sum(torch.mean(torch.sqrt(torch.sum((x_pred - x_true)**2, dim=3)), dim=[1,2]), dim=0)
