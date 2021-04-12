import time
import torch
import utils
import numpy as np
from importlib import import_module
import argparse
import train

parser = argparse.ArgumentParser(description='An-bert-Text-Classification')
parser.add_argument('--model', type=str, default='an-bert', help='choose a model')
args = parser.parse_args()



if __name__ == '__main__':
    dataset = 'THUCNews' 
    model_name = args.model
    x = import_module('model.' + model_name)
    config = x.Config(dataset)

    # make sure every out come same 
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print('Loadiing dataset......')
    train_data, dev_data, test_data = utils.bulid_dataset(config)
    train_iter = utils.build_iterator(train_data, config)
    dev_iter = utils.build_iterator(dev_data, config)
    test_iter = utils.build_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print('The time of preparing data:', time_dif)

    # model Train
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)


