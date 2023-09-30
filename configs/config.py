import argparse
import inspect
import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    train_epochs = 20
    batch_size = 128
    learning_rate = 0.002
    l2_regularization = 1e-6  # Weight decay degree
    learning_rate_decay = 0.99  # Learning rate decay degree

    word2vec_file = 'embedding/glove.6B.50d.txt'
    train_file = 'data/yelp/train.csv'
    valid_file = 'data/yelp/valid.csv'
    test_file = 'data/yelp/test.csv'
    model_file = 'model/best_model.pt'

    review_count = 10  # Maximum review count
    review_length = 40  # Maximum review length
    lowest_review_count = 2  # Reviews written by a user/item will be deleted if the amount is less than this value
    PAD_WORD = '<UNK>'

    kernel_count = 100
    kernel_size = 3
    dropout_prob = 0.5
    cnn_out_dim = 50  # CNN output dimension

    def __init__(self):
        # We can customize parameters through command line arguments.
        # For example:
        # python main.py --device cuda:0 --train_epochs 50
        # Get all attributes of this class
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        # Create an ArgumentParser object
        parser = argparse.ArgumentParser()
        # Add all attributes as arguments to the parser
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        # Parse the command line arguments and set attributes with their values
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        # Get all attributes of this class
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        # Concatenate all attributes and their values to a string
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
