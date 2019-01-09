import argparse
from utils import load_data, DATA_GRP
from dl_model import create_model, train_model, save_model
import logging as logger
import os
from torch import optim
import torch
from torch import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='data_dir',
                    help='set data directory for training/validation')
    parser.add_argument('--save_dir', dest='save_dir',
                    help='set directory to save checkpoints')
    parser.add_argument('--arch', dest='arch', default='vgg16',
                    help='choose pretrained model architecture')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001,
                    help='choose hyperparameter: learning rate')
    parser.add_argument('--hidden_units', nargs='+',type=int, default=[4096, 4096],
                    help='choose hyperparameter: number of hidden unites in the form unit1, unit2')
    parser.add_argument('--epochs', dest='epochs', type=int, default=4,
                    help='choose hyperparameter: number of epochs to train the network')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training')
    parsed_args = parser.parse_args()
    logger.info('loading data in to dataloader')
    img_datasets, img_dataloaders = load_data(parsed_args.data_dir)
    dataset_sizes = {x: len(img_datasets[x]) for x in DATA_GRP}
    # ----- printing some useful info about the data ----"
    for x in DATA_GRP:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))
    print("Classes: ")
    class_names = img_datasets[DATA_GRP[0]].classes
    #print(img_datasets[DATA_GRP[0]].classes)
    dataset_sizes = {x: len(img_datasets[x]) for x in DATA_GRP}
    print(dataset_sizes)
    # --- create model ( modify classifier) and train it
    logger.info('loading {} model and modeifying classifier to match the output layer'.format(parsed_args.arch))
    model = create_model(hidden_layers = parsed_args.hidden_units, output_layer = len(class_names), arch=parsed_args.arch)
    learning_rate = parsed_args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    logger.info('training {} model'.format(parsed_args.arch))
    model = train_model(img_dataloaders=img_dataloaders, dataset_sizes=dataset_sizes, model=model, criterion = criterion, 
                        optimizer = optimizer, epochs = parsed_args.epochs, is_gpu=parsed_args.gpu)
    logger.info('saving {} model'.format(parsed_args.arch))
    save_model(model, parsed_args.save_dir, img_datasets['train'].class_to_idx, hidden_layers = parsed_args.hidden_units, output_layer = len(class_names), arch=parsed_args.arch)
                
    