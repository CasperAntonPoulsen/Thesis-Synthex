import argparse


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument("--model-name", type=str)
    group.add_argument('--model-dir', type=str, help='model path')
    group.add_argument("--data-dir")
    group.add_argument("--epochs", type=int)
    group.add_argument("--gamma", type=float)
    group.add_argument("--learning-rate", type=float)
    group.add_argument("--batch-size", type=int)
    return parser

