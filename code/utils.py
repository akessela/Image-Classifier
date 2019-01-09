import torch
from torchvision import datasets, transforms
import os
from PIL import Image
from torch.autograd import Variable


__all__ = ['load_data']

DATA_GRP = ['train', 'valid', 'test']
# shift and normalize image
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SHORTEST_SIZE = 256
DESIRED_IMAGE_SIZE = 224
IMG_ROTATION = 30
data_transforms =  {
    # Data augmentation for traning set rotate by 30 degree, crop to 224, flip horizontally
    DATA_GRP[0]:transforms.Compose([transforms.RandomRotation(IMG_ROTATION),
                                    transforms.Resize(SHORTEST_SIZE),
                                    transforms.CenterCrop(DESIRED_IMAGE_SIZE),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN,STD) 
                                   ]),    
    #tranforms to both validation and test data (only resize and normalize)
    DATA_GRP[1]:transforms.Compose([ transforms.Resize(SHORTEST_SIZE),
                                    transforms.CenterCrop(DESIRED_IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN,STD) 
                                   ]),
                                           
     DATA_GRP[2]:transforms.Compose([ transforms.Resize(SHORTEST_SIZE),
                                    transforms.CenterCrop(DESIRED_IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(MEAN,STD) 
                                   ]),
}
def load_data(data_directory):
    img_datasets = {sub_dir : datasets.ImageFolder(os.path.join(data_directory, sub_dir),
                                                   transform=data_transforms[sub_dir]
                                                  ) for sub_dir in DATA_GRP
        
                   }
    img_dataloaders = {sub_dir:torch.utils.data.DataLoader(img_datasets[sub_dir], batch_size=64,
                                                           shuffle=True) for sub_dir in DATA_GRP
                      }
    return img_datasets, img_dataloaders

def image_loader(image_path, use_gpu=False):
    image = Image.open(image_path)
    image = data_transforms[DATA_GRP[2]](image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    if use_gpu:
        return image.cuda()
    return image
    


                       
