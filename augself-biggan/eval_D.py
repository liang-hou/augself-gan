import torch
from tqdm import tqdm
import numpy as np
import functools
import os
import inception_tf
import utils
import dnnlib
from sklearn.linear_model import LogisticRegression


def run_eval(config):
    # update config (see train.py for explanation)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config = utils.update_config_roots(config)
    config['skip_init'] = True
    config['no_optim'] = True
    config['num_epochs'] = 1
    config['num_samples'] = 50000
    device = 'cuda'

    model = __import__(config['model'])
    # G = model.Generator(**config).cuda()
    D_batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])
    D = model.Discriminator(**config).cuda()
    D.load_state_dict(torch.load(dnnlib.util.open_file_or_url(config['network'])), strict=False)
    D.eval()

    acc_list = []
    for _ in tqdm(range(config['repeat'])):
        loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size})
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(
                loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])
        
        train_data, train_label = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(pbar):
                if config['D_fp16']:
                    x, y = x.to(device).half(), y.to(device)
                else:
                    x, y = x.to(device), y.to(device)
                h = x
                for index, blocklist in enumerate(D.blocks):
                    for block in blocklist:
                        h = block(h)
                h = torch.sum(D.activation(h), [2, 3])
                train_data.append(h.cpu().numpy())
                train_label.append(y.cpu().numpy())
        train_data = np.vstack(train_data)
        train_label = np.hstack(train_label)

        print('train data', train_data.shape)
        print('train label', train_label.shape)

        loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size, 'train': False})
        # Which progressbar to use? TQDM or my own?
        if config['pbar'] == 'mine':
            pbar = utils.progress(
                loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(loaders[0])

        test_data, test_label = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(pbar):
                if config['D_fp16']:
                    x, y = x.to(device).half(), y.to(device)
                else:
                    x, y = x.to(device), y.to(device)
                h = x
                for index, blocklist in enumerate(D.blocks):
                    for block in blocklist:
                        h = block(h)
                h = torch.sum(D.activation(h), [2, 3])
                test_data.append(h.cpu().numpy())
                test_label.append(y.cpu().numpy())
        test_data = np.vstack(test_data)
        test_label = np.hstack(test_label)

        print('test data', test_data.shape)
        print('test label', test_label.shape)

        LR = LogisticRegression()
        LR.fit(train_data, train_label)
        acc = LR.score(test_data, test_label)
        acc_list.append(acc)
    
    if config['repeat'] > 1:
        print('ACC mean: {}, std: {}'.format(np.mean(acc_list), np.std(acc_list)))
    else:
        print('ACC: {}'.format(np.mean(acc_list)))


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    run_eval(config)


if __name__ == '__main__':
    main()
