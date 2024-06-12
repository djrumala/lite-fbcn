import argparse
import numpy as np

from data_load import load_pickle, create_training_data, get_XandY
from utils import model_evaluate

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
    parser.add_argument('--model', type=str, default='mobilenetv1', help='Name of CNN model to be evaluated')
    parser.add_argument('--sampling_name', type=str, default='hold_out', help='Which dataset to use: hold_out for hold out data, and fold1 - fold5 for cross validation')
    parser.add_argument('--input_shape', type=tuple, default=(224, 224, 3), help='Input shape for the model')
    parser.add_argument('--num_class', type=int, default=5, help='Number of class for the categorization')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epoch to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Optimizer to train the model')
    parser.add_argument('--opt_nesterov', type=bool, default=True, help='Optimizer Nesterov (specially for SGD)')

    args = parser.parse_args()
    
    print(args.model)
    if 'hold_out' in args.sampling_name: #hold-out dataset evaluation
        CATEGORIES = ["CN", "NEO", "CVA", "NDD", "INF"]
        test_combined = create_training_data(CATEGORIES=CATEGORIES, DATADIR= f'../data/holdout/')
        test_X, test_Y =  get_XandY(test_combined)
    else: #cross-validation dataset evaluation
        fold = args.sampling_name[1]
        
        test_X = load_pickle("test", "X", fold)
        test_Y = load_pickle("test", "Y", fold)

    model_evaluate(test_X, test_Y, cnn_model=args.model, sampling_name=args.sampling_name, input_shape=args.input_shape, num_class=args.num_class, num_ep=args.num_epoch, batch_size=args.batch_size, learning_rate=args.learning_rate, opt_nesterov=args.opt_nesterov)
    
if __name__ == "__main__":
    main()