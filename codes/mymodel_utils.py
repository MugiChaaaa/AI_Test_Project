### Import Other Libraries
import math


def get_cnn_channels(initial_ch: int = 32, length: int = 2, multiplier: float = 2.0) -> tuple:
    """
    Get the number of channels for the CNN model.
    :param initial_ch: Initial number of channels. Default is 32.
    :param length: Number of layers. Default is 2.
    :param multiplier: Multiplier for the number of channels. Default is 2.0.
    :return: _cnn_ch: tuple of number of channels for the CNN model.
    """
    _cnn_ch = (initial_ch, )
    for i in range(length - 1):
        _cnn_ch = _cnn_ch + (int(multiplier * _cnn_ch[-1]), )
    return _cnn_ch


def get_cnn_feature_size(initial_features: int = 2048, length: int = 2, output_features: int = 10) -> tuple:
    """
    Get the feature size for the CNN model.
    :param initial_features: Initial number of features. Default is 2048.
    :param length: Number of layers. Default is 2.
    :param output_features: Number of output features. Default is 10.
    :return: _feature_size: tuple of feature size for the CNN model. 'output_features' is not included.
    """
    _feature_size = (initial_features, )
    y1 = math.log2(initial_features)
    y2 = math.log2(output_features)
    for i in range(length - 1):
        m = (length - i - 1) / length
        n = 1 - m
        exp_value = int(y1 * m + y2 * n)
        _feature_size = _feature_size + (int(pow(2, exp_value)), )
    return _feature_size