import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument("--path", type=str, default="/root/NSP/data/1.csv", help="path of your dataset")
    parser.add_argument("--repetitions", type=int, default=10, help="repetition times of your experiment")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="max epochs for training")
    parser.add_argument("--hidden", type=int, default=36, help="hiddensize of lstm")
    parser.add_argument("--trainsize", type=int, default=8, help="size of training set")
    parser.add_argument("--layers", type=int, default=4, help="layer sizes of lstm")



    return parser.parse_known_args()[0]
