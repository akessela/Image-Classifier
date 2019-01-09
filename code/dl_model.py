from torchvision import models
from torch import nn
from torch import optim
from collections import OrderedDict
from utils import DATA_GRP, image_loader
import logging as logger
import torch
import os
import json
import numpy as np

def create_model(hidden_layers, output_layer, arch):
    model = getattr(models, arch)(pretrained=True)
    if arch in ['vgg13', 'vgg16']:
        in_features_size = model.classifier[0].in_features
    elif arch in ['resnet18']:
        in_features_size = model.fc.in_features
    for param in model.parameters():
        param.requires_grad = False
    #build the network 
    network =  [('0', nn.Linear(in_features_size, hidden_layers[0]))]
    network.append(('1', nn.ReLU()))
    network.append(('2', nn.Dropout(0.1)))
    for i, (a,b) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
        network.append((str(i + 3), nn.Linear(a, b)))
        network.append((str(i + 4), nn.ReLU()))  
        network.append((str(i + 5), nn.Dropout(0.1)))
    u = len(network) + 1
    network.append((str(u), nn.Linear(hidden_layers[-1], output_layer)))
    network.append((str(u + 1), nn.LogSoftmax(dim=1)))
    classifier = nn.Sequential(OrderedDict(network))
    model.classifier = classifier
    return model
def train_model(img_dataloaders, dataset_sizes, model, criterion, optimizer, epochs, is_gpu):
    steps = 0
    if is_gpu:
        model.to('cuda')
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in iter(img_dataloaders[DATA_GRP[0]]):
            steps += 1
            if is_gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            # forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item() 
        print("Epoch: {}/{} ...".format(e + 1, epochs),
                  " Average Training Loss: {:.4f}".format(running_loss/dataset_sizes['train']))
        compute_validation_loss(model, img_dataloaders[DATA_GRP[1]], e + 1, dataset_sizes['valid'], is_gpu, criterion)
    return model

def compute_validation_loss(model, validation_dataloader, which_epoch, val_dataset_size, is_gpu, criterion):
     # TODO: Do validation on the test set
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for images, labels in validation_dataloader:
            if is_gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print('Accuracy of the network after  epoch {} on the validation images: {:4f}%'.format(which_epoch, (100 * correct / total)))
    print("validation average Loss: {:.4f}".format(val_loss/val_dataset_size))
        
def save_model(model, save_dir, class_to_idx, hidden_layers, output_layer, arch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state = {
        'static_dict':model.state_dict(),
        'hidden_layers':hidden_layers,
        'output_layer' : output_layer,
        'arch' : arch,
        'class_to_idx':class_to_idx
    }       
    path = os.path.join(save_dir,'checkpoint.pth')
    print('saving model to {}'.format(path))
    torch.save(state, path)
     
def reload_model(path, is_gpu):
    if is_gpu:
        state = torch.load(path)
    else:
        state = torch.load(path, map_location=lambda storage, loc: storage)
    hidden_layers = state['hidden_layers']
    output_layer= state['output_layer']
    arch = state['arch']
    model = create_model(hidden_layers, output_layer, arch)
    model.load_state_dict(state['static_dict'])
    class_to_idx = state['class_to_idx']
    return model, class_to_idx


def predict(image_path, model, class_to_idx, topk, category_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    np_image = image_loader(image_path)
    with torch.no_grad():
        outputs = model(np_image)
    probs, class_idxs = outputs.topk(topk)
    np_idxs = class_idxs.numpy()
    idxs = [np_idxs.item(i) for i in range(np_idxs.shape[1])]
    classes = [str(idx_to_class[idx]) for idx in idxs ]
    pro = np.exp(probs.detach().numpy()[0])
    cat_to_name = None
    if category_name:
        with open(category_name, 'r') as f:
            cat_to_name = json.load(f)
    if topk ==1 :
        if cat_to_name:
            print('This flower is predicted to be {} with probability of {:.4f}'.format(cat_to_name[classes[0]], pro[0]))
        else:
           print('This flower is predicted to be  class {} with probability of {:.4f}'.format(classes[0], pro[0])) 
    else:
        if cat_to_name:
            for c in classes:
                print('class', 'flower_name', 'probability')
                print(c, cat_to_name[c[0]], pro[0])
        else:
            print('top {} likely classes this flower belongs too:'.format(topk))
            print(classes)
            print('with corresponding probabilities')
            print(pro)
            
        
        
            
                      
        






    
    