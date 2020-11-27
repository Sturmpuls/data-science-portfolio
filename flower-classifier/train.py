import argparse
import time
import torch

from classifier import ImageClassifier
from pathlib import Path
from utils import ImageLoader, labels_from_json


def main():
    description = 'Trains a Neural Network on a dataset and saves the trained model to a file.'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('data_dir',
                        type=str,
                        help='Working directory containing the data.')
    parser.add_argument('--save_dir',
                        dest='save_dir',
                        metavar='DIR',
                        type=str,
                        help='Folder where the trained model will be stored.')
    parser.add_argument('--name',
                        dest='filename',
                        metavar='FN',
                        type=str,
                        help='Filename for the trained model.')
    parser.add_argument('--epochs',
                        dest='epochs',
                        metavar='E',
                        default=10,
                        type=int,
                        help='Number of cycles for which the model will run.')
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        metavar='LR',
                        default=0.001,
                        type=float,
                        help='Determines how fast the model learns. Default value usually works pretty well.')
    parser.add_argument('--arch',
                        dest='arch',
                        default='vgg19',
                        type=str,
                        choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152',],
                        help='Pretrained Network that will be the core of the model.')
    parser.add_argument('--hidden_units',
                        dest='hidden_units',
                        metavar='N',
                        default=2048,
                        type=int,
                        help='Number of neurons in the hidden layer.')
    parser.add_argument('--gpu',
                        dest='gpu',
                        default=False,
                        type=bool,
                        nargs='?',
                        const=True,
                        help='Use GPU for training.')
    parser.add_argument('--dropout',
                        dest='dropout',
                        metavar='P',
                        default=0.2,
                        type=float,
                        help='Probability that a neuron in the hidden layers is deactivated during training. Can improve model results.')
    parser.add_argument('--n_classes',
                        dest='n_classes',
                        default=102,
                        type=int,
                        help='Number of feature classes in the dataset.')
    parser.add_argument('--retrain',
                        dest='retrain',
                        default=False,
                        type=bool,
                        help='If set a checkpoint with the same name will be ignored and overwritten. If not set (defaults to False) the checkpoint will be loaded and used for training.')
    
    # Evaluate user input
    args = parser.parse_args()
    args.save_dir = args.save_dir if args.save_dir else ''
    args.filename = args.filename if args.filename else f'{args.arch}_checkpoint.pth'
    file_path = Path(args.save_dir) / args.filename
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
    
    # Create the model
    model = ImageClassifier(device)

    if Path(file_path).exists() and not args.retrain:
        model.load(file_path)
        print(f'Existing checkpoint "{file_path}" loaded.')
    else:
        model.initialize(args.arch, args.n_classes, learningrate=args.learning_rate,
                         n_hidden_units=args.hidden_units, p_dropout=args.dropout)
        
    # Load data
    IL = ImageLoader(args.data_dir)
    dataloader = IL.load()
    
    # Train model
    model.load_indexes_labels(labels_from_json(), IL.class_to_idx)
    for e in range(1, args.epochs+1):
        start = time.time()
        train_loss = model.train(dataloader['train'])
        valid_loss, valid_accuracy = model.validate(dataloader['valid'])
        stop = time.time()
        print('Epoch:', e,'Training Loss:', train_loss, 'Validation Loss:', valid_loss, 'Validation Accuracy:', valid_accuracy, 'Duration:', stop - start)

    # Save results
    print(f'Saving checkpoint "{file_path}"')
    model.save(file_path)

if __name__ == '__main__':
    main()