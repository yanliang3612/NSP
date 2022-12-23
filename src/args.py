import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # train

    parser.add_argument("--repetitions", type=int, default=20, help="repetition times of your experiment")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="max epochs for training")
    parser.add_argument("--hidden", type=int, default=96, help="hiddensize of lstm")
    parser.add_argument("--trainsize", type=int, default=8, help="size of training set")
    parser.add_argument("--layers", type=int, default=4, help="layer sizes of lstm")
    parser.add_argument("--width", type=int, default=96, help="width of BIN")
    parser.add_argument("--numargmen", type=int, default=24, help="number of synthetic samples to try oversampling")

    #trick
    parser.add_argument("--parside", type=bool, default=False, help="if using parside feature")
    parser.add_argument("--smote", type=bool, default=False, help="if using smote algorithm to try data augmentation")
    parser.add_argument("--oversampling", type=bool, default=True, help="if using oversampling to try data augmentation")
    parser.add_argument("--selftraining", type=bool, default=False,help="if using selftraining to try data augmentation")

    return parser.parse_known_args()[0]
