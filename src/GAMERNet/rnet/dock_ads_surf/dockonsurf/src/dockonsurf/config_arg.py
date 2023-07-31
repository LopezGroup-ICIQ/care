"""Configures the command-line arguments the programs can be called with"""
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='input file to read the calculation details.',
                        required=True)
    parser.add_argument('-f', '--foreground', default=False,
                        action='store_true', help='runs in foreground mode.')
    # parser.add_argument('--debug',  # TODO Implement
    #                     help='Turns on debug verbosity')
    return parser.parse_args()
