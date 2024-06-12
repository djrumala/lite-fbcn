import argparse
import os

from data_load import load_pickle
from utils import model_build_train

def valid_limit(value):
    try:
        start, end = map(int, value.split('-'))
        if start < 0 or end < start:
            raise argparse.ArgumentTypeError("Invalid limit format. Must be 'start-end' with start >= 0 and end >= start.")
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid limit format. Must be 'start-end' with start and end as integers.")

def valid_size(value):
    try:
        size = int(value)
        if size <= 0:
            raise argparse.ArgumentTypeError("Invalid size. Must be a positive integer.")
        return size
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid size. Must be a positive integer.")

def main():
    parser = argparse.ArgumentParser(description='Process images with specified parameters.')
    
    #for model traning
    parser.add_argument('--input_shape', type=tuple, default=(224, 224, 3), help='Input shape for the model')
    parser.add_argument('--num_class', type=int, default=5, help='Number of class for the categorization')
    parser.add_argument('--num_epoch', type=int, default=500, help='Number of epoch to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Optimizer to train the model')
    parser.add_argument('--opt_nesterov', type=bool, default=True, help='Optimizer Nesterov (for SGD)')
    parser.add_argument('--lr_reduce', type=bool, default=True, help='ReduceonPlateu usage (False/True)')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.1, help='Factor value for ReduceonPlateu')
    parser.add_argument('--lr_min', type=float, default=1e-4, help='Factor value for ReduceonPlateu')
    parser.add_argument('--lr_reduce_patience', type=int, default=50, help='Number of epoch to wait for ReduceonPlateu')

    #for data loading
    parser.add_argument('--round', type=str, default='R1', help='Sampling round')
    args = parser.parse_args()

    fold = args.round[1]

    train_X = load_pickle("train", "X", fold)
    train_Y = load_pickle("train", "Y", fold)
    valid_X = load_pickle("valid", "X", fold)
    valid_Y = load_pickle("valid", "Y", fold)

    model_build_train(train_X, train_Y, valid_X, valid_Y, round_label=args.round, input_shape=args.input_shape, num_class=args.num_class, num_ep=args.num_epoch, batch_size=args.batch_size, learning_rate=args.learning_rate, opt_nesterov=args.opt_nesterov, lr_reduce=args.lr_reduce, lr_reduce_factor= args.lr_reduce_factor, lr_reduce_patience= args.lr_reduce_patience, lr_min=args.lr_min)
    
    print(f'Congrats! Training done')
if __name__ == "__main__":
    main()