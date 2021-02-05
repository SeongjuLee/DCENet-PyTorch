
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:31:32 2019
@author: cheng
"""
import os
import sys
import time
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoints', filename='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.filename = filename
        os.makedirs(path, exist_ok=True)

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, self.filename))
        self.val_loss_min = val_loss


def get_errors(groundtruth, predictions):
    _, num_pred, pred_seq, _ = predictions.shape

    # Calculate the corresponding errors
    errors = get_evaluation(groundtruth, predictions, num_pred)
    # print('\nErrors by ADE and FDE\n', np.array_str(errors[0:2, 2], precision=2, suppress_small=True))
    
    print("Avearge displacement error: %.2f [m]"%(errors[0, 2]))
    print("Final displacement error: %.2f [m]" % (errors[1, 2]))
    print("Hausdorff distance: %.2f [m]" % (errors[2, 2]))
    print("Speed deviation: %.2f [m/s]" % (errors[3, 2]))
    print("Heading error: %.2f [°]" % (errors[4, 2]))

    return errors
    

def get_evaluation(groundtruth, predictions, num_pred, scale=1.0):
    # Evaluation
    evaluations = np.zeros([len(predictions), num_pred, 5])
    for i, user_gt in enumerate(groundtruth):
        user_preds = predictions[i]
        for j, user_pred in enumerate(user_preds):
            evaluations[i, j, :] = get_eva_values(user_gt[:, 2:4]*scale, user_pred*scale)
    # Compute the average errors across all users and all predictions
    mean_evaluations = np.reshape(evaluations, [-1, 5])
    mean_errors = np.mean(mean_evaluations, axis=0)
    mean_std = np.std(mean_evaluations, axis=0)
    # Comput the minimum errors across all users for the best prediction
    min_evaluations = np.min(evaluations, axis=1)
    min_errors = np.mean(min_evaluations, axis=0)
    min_std = np.std(min_evaluations, axis=0)
    # Save the evaluation results
    errors = np.concatenate((np.reshape(mean_errors, [-1, 1]), 
                             np.reshape(mean_std, [-1, 1]),
                             np.reshape(min_errors, [-1, 1]),
                             np.reshape(min_std, [-1, 1])), axis=1)
    return errors


def get_eva_values(y_t, y_p):
    '''
    y_t: 2d numpy array for true trajectory. Shape: steps*2
    y_p: 2d numpy array for predicted trajectory. Shape: steps*2
    '''    
    Euclidean = get_euclidean(y_t, y_p)    
    last_disp = get_last_disp(y_t, y_p)   
    Hausdorff = get_hausdorff(y_t, y_p)    
    speed_dev = get_speeddev(y_t, y_p)    
    heading_error = get_headerror(y_t, y_p)
    eva_values = [Euclidean, last_disp, Hausdorff, speed_dev, heading_error]    
    return eva_values
   
def get_euclidean(y_true, y_prediction):
    Euclidean = np.linalg.norm((y_true - y_prediction), axis=1)
    Euclidean = np.mean(Euclidean)
    return Euclidean

def get_last_disp(y_true, y_prediction):
    last_disp = np.linalg.norm((y_true[-1, :] - y_prediction[-1, :]))
    return last_disp
        
def get_hausdorff(y_true, y_prediction):
    '''
    Here is the directed Hausdorff distance, but it computes both directions and output the larger value
    '''
    Hausdorff = max(directed_hausdorff(y_true, y_prediction)[0], directed_hausdorff(y_prediction, y_true)[0])
    return Hausdorff
    
def get_speeddev(y_true, y_prediction):
    if len(y_true) == 1:
        return 0
    else:       
        speed_dev = 0.0
        for t in range(len(y_true)-1):
            speed_t = np.linalg.norm(y_true[t+1] - y_true[t])       
            speed_p = np.linalg.norm(y_prediction[t+1] - y_prediction[t])
            speed_dev += abs(speed_t - speed_p)
        speed_dev /=  (len(y_true)-1)
        return speed_dev     

def get_headerror(y_true, y_prediction):
    if len(y_prediction) == 1:
        return 0
    else:
        heading_error = 0.0
        for t in range(len(y_true)-1):
            xcoor_t = y_true[t+1, 0] - y_true[t, 0]
            ycoor_t = y_true[t+1, 1] - y_true[t, 1]
            angle_t = np.arctan2(ycoor_t, xcoor_t)
            xcoor_p = y_prediction[t+1, 0] - y_prediction[t, 0]
            ycoor_p = y_prediction[t+1, 1] - y_prediction[t, 1]
            angle_p = np.arctan2(ycoor_p, xcoor_p)
            angle = np.rad2deg((abs(angle_t - angle_p)) % (np.pi))
            heading_error += angle
        heading_error /= len(y_true)-1
        return heading_error

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def input2tensor(traj, occu, device):
    traj = torch.tensor(traj, dtype=torch.float)
    occu = torch.tensor(occu, dtype=torch.float).permute(0, 1, 4, 2, 3)

    return traj.to(device), occu.to(device)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']