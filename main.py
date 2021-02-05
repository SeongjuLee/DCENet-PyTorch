'''Train DCENet with PyTorch'''
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import os
import json
import neptune
import argparse
import numpy as np

from loader import *
from utils.plots import *
from utils.utils import *
from utils.collision import *
from utils.datainfo import DataInfo
from utils.ranking import gauss_rank

from models import DCENet
from loss import DCENetLoss



def main():

    # ================= Arguments ================ #
    parser = argparse.ArgumentParser(description='PyTorch Knowledge Distillation')
    parser.add_argument('--gpu', type=str, default="4", help='gpu id')
    parser.add_argument('--config', type=str, default="config", help='.json')
    args = parser.parse_args()

    # ================= Device Setup ================ #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================= Config Load ================ #
    with open('config/' + args.config) as config_file:
        config = json.load(config_file)

    # ================= Neptune Setup ================ #
    if config['neptune']:
        neptune.init('seongjulee/DCENet', api_token=config["neptune_token"])   # username/project-name, api_token=token from neptune
        neptune.create_experiment(name='EXP', params=config)  # name=project name (anything is ok), params=parameter list (json format)
        neptune.append_tag(args.config) # neptune tag (str or string list)

    # ================= Model Setup ================ #    
    model = nn.DataParallel(DCENet(config)).to(device) if len(args.gpu.split(',')) > 1 else DCENet(config).to(device)
    
    # ================= Loss Function ================ #
    criterion = DCENetLoss(config)

    # ================= Optimizer Setup ================ #
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-6, amsgrad=False)

    # ================= Data Loader ================ #
    datalist = DataInfo()
    train_datalist = datalist.train_merged
    print('Train data list', train_datalist)

    test_datalist = datalist.train_biwi
    print('Test data list', test_datalist)

    np.random.seed(10)
    offsets, traj_data, occupancy = load_data(config, train_datalist, datatype="train")
    trainval_split = np.random.rand(len(offsets)) < config['split']

    train_x = offsets[trainval_split, :config['obs_seq'] - 1, 4:6]
    train_occu = occupancy[trainval_split, :config['obs_seq'] - 1, ..., :config['enviro_pdim'][-1]]
    train_y = offsets[trainval_split, config['obs_seq'] - 1:, 4:6]
    train_y_occu = occupancy[trainval_split, config['obs_seq'] - 1:, ..., :config['enviro_pdim'][-1]]

    val_x = offsets[~trainval_split, :config['obs_seq'] - 1, 4:6]
    val_occu = occupancy[~trainval_split, :config['obs_seq'] - 1, ..., :config['enviro_pdim'][-1]]
    val_y = offsets[~trainval_split, config['obs_seq'] - 1:, 4:6]
    val_y_occu = occupancy[~trainval_split, config['obs_seq'] - 1:, ..., :config['enviro_pdim'][-1]]

    print("%.0f trajectories for training\n %.0f trajectories for valiadation" %(train_x.shape[0], val_x.shape[0]))

    test_offsets, test_trajs, test_occupancy = load_data(config, test_datalist, datatype="test")
    test_x = test_offsets[:, :config['obs_seq'] - 1, 4:6]
    test_occu = test_occupancy[:, :config['obs_seq'] - 1, ..., :config['enviro_pdim'][-1]]
    last_obs_test = test_offsets[:, config['obs_seq'] - 2, 2:4]
    y_truth = test_offsets[:, config['obs_seq'] - 1:, :4]
    xy_truth = test_offsets[:, :, :4]

    print('test_trajs', test_trajs.shape)

    print("%.0f trajectories for testing" % (test_x.shape[0]))

    train_dataset = TrajDataset(x=train_x, x_occu=train_occu, y=train_y, y_occu=train_y_occu, mode='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    val_dataset = TrajDataset(x=val_x, x_occu=val_occu, y=val_y, y_occu=val_y_occu, mode='val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    
    # test_dataset = TrajDataset(x=test_x, x_occu=test_occu, y=y_truth, y_occu=None, mode='test')
    # test_loader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # ================= Training Loop ================ #
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True, filename=args.config.split('/')[-1].replace('.json', '.pth'))
    for epoch in range(config['max_epochs']):
        train_one_epoch(config, epoch, device, model, optimizer, criterion, train_loader)
        val_loss = evaluate(config, device, model, optimizer, criterion, val_loader)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # ================= Test ================ #
    model.load_state_dict(torch.load(os.path.join('checkpoints', args.config.split('/')[-1].replace('.json', '.pth'))))
    model.eval()
    with torch.no_grad():
        test_x, test_occu = input2tensor(test_x, test_occu, device)
        x_latent = model.encoder_x(test_x, test_occu)
        predictions = []
        for i, x_ in enumerate(x_latent):
            last_pos = last_obs_test[i]
            x_ = x_.view(1, -1)
            for i in range(config['num_pred']):
                y_p = model.decoder(x_, train=False)
                y_p_ = np.concatenate(([last_pos], np.squeeze(y_p.cpu().numpy())), axis=0)
                y_p_sum = np.cumsum(y_p_, axis=0)
                predictions.append(y_p_sum[1:, :])

    predictions = np.reshape(predictions, [-1, config['num_pred'], config['pred_seq'], 2])

    print('Predicting done!')
    print(predictions.shape)
    plot_pred(xy_truth, predictions)
    # Get the errors for ADE, DEF, Hausdorff distance, speed deviation, heading error
    print("\nEvaluation results @top%.0f" % config['num_pred'])
    errors = get_errors(y_truth, predictions)
    check_collision(y_truth)

    ## Get the first time prediction by g
    ranked_prediction = []
    for prediction in predictions:
        ranks = gauss_rank(prediction)
        ranked_prediction.append(prediction[np.argmax(ranks)])
    ranked_prediction = np.reshape(ranked_prediction, [-1, 1, config['pred_seq'], 2])
    print("\nEvaluation results for most-likely predictions")
    ranked_errors = get_errors(y_truth, ranked_prediction)


# Function for one epoch training
def train_one_epoch(config, epoch, device, model, optimizer, criterion, loader):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_total, train_loss = 0, 0

    for batch_idx, (x, x_occu, y, y_occu) in enumerate(loader):
        x, x_occu, y, y_occu = x.to(device), x_occu.to(device), y.to(device), y_occu.to(device)
        optimizer.zero_grad()

        y_pred, mu, log_var = model(x, x_occu, y, y_occu, train=True)
        loss = criterion(mu, log_var, y_pred, y)

        loss.backward()
        optimizer.step()

        # train_ade += ade * x.size(0)
        # train_fde += fde * x.size(0)
        train_total += x.size(0)
        train_loss += loss.item() * x.size(0)

        if config['neptune']:
            # neptune.log_metric('train_batch_ADE', ade)
            # neptune.log_metric('train_batch_FDE', fde)
            neptune.log_metric('train_batch_Loss', loss.item())

        # progress_bar(batch_idx, len(loader), 'Lr: %.4e | Loss: %.3f | ADE[m]: %.3f | FDE[m]: %.3f'
        #     % (get_lr(optimizer), train_loss / train_total, train_ade / train_total, train_fde / train_total))
        progress_bar(batch_idx, len(loader), 'Lr: %.4e | Loss: %.3f' % (get_lr(optimizer), train_loss / train_total))


# Function for validation
@torch.no_grad()
def evaluate(config, device, model, optimizer, criterion, loader):
    model.eval()
    # eval_ade, eval_fde, eval_total = 0, 0, 0
    eval_total, eval_loss = 0, 0

    for batch_idx, (x, x_occu, y, y_occu) in enumerate(loader):
        x, x_occu, y, y_occu = x.to(device), x_occu.to(device), y.to(device), y_occu.to(device)

        y_pred, mu, log_var = model(x, x_occu, y, y_occu, train=True)
        loss = criterion(mu, log_var, y_pred, y)

        eval_total += x.size(0)
        eval_loss += loss.item() * x.size(0)

        progress_bar(batch_idx, len(loader), 'Lr: %.4e | Loss: %.3f' % (get_lr(optimizer), eval_loss / eval_total))
        # progress_bar(batch_idx, len(loader), 'Lr: %.4e | ADE[m]: %.3f | FDE[m]: %.3f'
            # % (get_lr(optimizer), eval_ade / eval_total, eval_fde / eval_total))
    
    if config['neptune']:
        neptune.log_metric('val_Loss', eval_loss / eval_total)

    #     neptune.log_metric('{}_ADE'.format(loader.dataset.mode), eval_ade / eval_total)
    #     neptune.log_metric('{}_FDE'.format(loader.dataset.mode), eval_fde / eval_total)
    
    return eval_loss / eval_total


if __name__ == "__main__":
    main()
