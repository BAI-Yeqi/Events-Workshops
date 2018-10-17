import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    args.config = '/data/workspace/yeqi/projects/RNN4REC/GRU4REC/Code/v2/configs/config.json'

    return args


