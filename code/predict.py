import argparse
from utils import image_loader
from dl_model import reload_model, predict
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='input',
                    help='image full path to predict')
    parser.add_argument(dest='checkpoint',
                    help='checkpoint path to save model')
    parser.add_argument('--top_k', dest='top_k',type=int, default=1,
                    help='top k most likely classes')
    parser.add_argument('--category_names', dest='category_names',
                    help='category to class names JSON file ')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training')
    args = parser.parse_args()
    image = image_loader(args.input)
    model, class_to_idx = reload_model(args.checkpoint, args.gpu) 
    predict(args.input, model, class_to_idx, args.top_k, args.category_names)
    # python predict.py flowers/train/1/image_06735.jpg  model_checkpoint/checkpoint.pth
    
    
    
    