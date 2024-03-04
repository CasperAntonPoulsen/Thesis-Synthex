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

def add_nbia_args(parser: argparse.ArgumentParser):
    """NBIA downloader arguments"""

    group = parser.add_argument_group('nbia', 'nbia download configuration')
    group.add_argument("--collection-name", type=str)
    group.add_argument("--output-dir", type=str)
    group.add_argument("--api-url", type=str, default="")
    return parser

def add_deepdrr_data_prep_args(parser: argparse.ArgumentParser):
    """Data prep for DeepDRR"""
    return parser

def add_classifiers_data_prep_args(parser: argparse.ArgumentParser):
    """Data prep for classifiers"""
    return parser

def add_deepdrr_args(parser: argparse.ArgumentParser):
    """DeepDRR arguments"""
    return parser