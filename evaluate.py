'''Evaluate DCENet with PyTorch'''
# from __future__ import print_function

import torch
import torch.nn as nn

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


def main():

    parser = argparse.ArgumentParser(description='DCENet PyTorch Implementation')
    parser.add_argument('--gpu', type=str, default="4", help='gpu id')
    parser.add_argument('--config', type=str, default="config", help='json file path')
    parser.add_argument('--resume-name', type=str, default="checkpoint.pth", help='checkpoint file path')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open('config/' + args.config) as config_file:
        config = json.load(config_file)

    if config['neptune']:
        neptune.init('seongjulee/DCENet', api_token=config["neptune_token"])   # username/project-name, api_token=token from neptune
        neptune.create_experiment(name='EXP', params=config)  # name=project name (anything is ok), params=parameter list (json format)
        neptune.append_tag(args.config) # neptune tag (str or string list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================= Model Setup ================ #    
    model = nn.DataParallel(DCENet(config)).to(device) if len(args.gpu.split(',')) > 1 else DCENet(config).to(device)

    # ================= Data Loader ================ #
    datalist = DataInfo()
    test_datalist = datalist.train_biwi
    print('Test data list', test_datalist)

    test_offsets, test_trajs, test_occupancy = load_data(config, test_datalist, datatype="test")
    test_x = test_offsets[:, :config['obs_seq'] - 1, 4:6]
    test_occu = test_occupancy[:, :config['obs_seq'] - 1, ..., :config['enviro_pdim'][-1]]
    last_obs_test = test_offsets[:, config['obs_seq'] - 2, 2:4]
    y_truth = test_offsets[:, config['obs_seq'] - 1:, :4]
    xy_truth = test_offsets[:, :, :4]

    print('test_trajs', test_trajs.shape)

    print("%.0f trajectories for testing" % (test_x.shape[0]))
    
    # Test
    model.load_state_dict(torch.load(os.path.join('checkpoints', args.resume_name)))
    model.eval()
    torch.manual_seed(10)
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


if __name__ == "__main__":
    main()
