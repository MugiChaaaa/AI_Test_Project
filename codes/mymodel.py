### Import Torch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.models as models
import torchvision.ops as ops

### Import Custom Libraries
import codes.mymodel_utils as mymu


class My2hl(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the model. The model is a simple feedforward neural network with 2 hidden layers.
        """
        super(My2hl, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. The input is reshaped to a 2D tensor and passed through the layers.
        :param x: Input tensor of shape (batch_size, 1, 28, 28).
        :return: _x: Output tensor of shape (batch_size, 10).
        """
        _x = x.view(-1, 28 * 28)
        _x = self.fc1(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc2(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc3(_x)
        _x = nnf.log_softmax(_x, dim=1)
        return _x


class My3hl(nn.Module):
    def __init__(self) -> None:
        """
        Initialize the model. The model is a simple feedforward neural network with 3 hidden layers.
        """
        super(My3hl, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. The input is reshaped to a 2D tensor and passed through the layers.
        :param x: Input tensor of shape (batch_size, 1, 28, 28).
        :return: _x: Output tensor of shape (batch_size, 10).
        """
        _x = x.view(-1, 28 * 28)
        _x = self.fc1(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc2(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc3(_x)
        _x = nnf.sigmoid(_x)
        _x = self.fc4(_x)
        _x = nnf.log_softmax(_x, dim=1)
        return _x


class CNN2Conv(nn.Module):
    def __init__(self, input_size: None | tuple[int, int, int, int] = None, output_size: None | int = None) -> None:
        """
        Initialize the model. The model is a simple convolutional neural network with 2 convolutional layers and 2 fully connected layers.
        :param input_size: (batch_size, channels, height, width). Input size of the model. Default is None. If None, error occurs.
        :param output_size: Output size of the model. Default is None. If None, the output size is set to 10.
        """
        if input_size is None:
            raise ValueError("param 'input_size' cannot be None")
        else:
            _input_size = input_size

        if output_size is None:
            _output_size = 10
        else:
            _output_size = output_size

        ### Assume that CNN channels double(*2) for each layer, and layers are 2 here.
        self.pool_kernel = 2
        _cnn_ch = mymu.get_cnn_channels(16, 2) ## (16, 32) when using cifar10 32 * 32.
        _channels = _input_size[1]
        _H = _input_size[2] ## height. 32 when using cifar10.
        _W = _input_size[3] ## width. 32 when using cifar10.
        _H_linear: int = _H // pow(self.pool_kernel, 2)
        _W_linear: int = _W // pow(self.pool_kernel, 2)
        self.features_linear:tuple = mymu.get_cnn_feature_size(_cnn_ch[-1] * _H_linear * _W_linear, 2, _output_size) ## (2048, 128) when using cifar10.

        super(CNN2Conv, self).__init__()
        self.conv1 = nn.Conv2d(_input_size[1], _cnn_ch[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(_cnn_ch[0], _cnn_ch[1], kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.features_linear[0], self.features_linear[1])
        self.fc2 = nn.Linear(self.features_linear[1], _output_size)
        self.flatten = nn.Flatten()

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. The input is passed through the convolutional layers and then through the fully connected layers.
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: _x: Output tensor of shape (batch_size, 10).
        """
        _x = self.conv1(x)
        _x = nnf.relu(_x)
        _x = nnf.max_pool2d(_x, kernel_size=self.pool_kernel)
        _x = self.conv2(_x)
        _x = nnf.relu(_x)
        _x = nnf.max_pool2d(_x, kernel_size=self.pool_kernel)

        _x = self.flatten(_x)
        _x = self.fc1(_x)
        _x = nnf.relu(_x)
        _x = self.fc2(_x)
        return _x


class CNN3linear(nn.Module):
    def __init__(self, input_size: None | tuple[int, int, int, int] = None, output_size: None | int = None, conv_num: int | None = None) -> None:
        """
        Initialize the model. The model is a simple convolutional neural network with n-convolutional layers and 3 fully connected layers.
        :param input_size: (batch_size, channels, height, width). Input size of the model. Default is None. If None, error occurs.
        :param output_size: Output size of the model. Default is None. If None, the output size is set to 10.
        :param conv_num: Number of convolutional layers. Default is None. If None, the number of convolutional layers is set to 3.
        """
        if input_size is None:
            raise ValueError("param 'input_size' cannot be None")
        else:
            _input_size = input_size

        if output_size is None:
            _output_size = 10
        else:
            _output_size = output_size

        if conv_num is None:
            _conv_num = 3
        else:
            _conv_num = conv_num

        ### Assume that CNN channels double(*2) for each layer, and linear layers are 3 here.
        self.pool_kernel = 2
        _linear_layers = 3
        _cnn_ch = mymu.get_cnn_channels(16, _conv_num) ## (16, 32, 64, ...) when using cifar10 32 * 32.
        _channels = _input_size[1]
        _H = _input_size[2] ## height. 32 when using cifar10.
        _W = _input_size[3] ## width. 32 when using cifar10.
        _H_linear: int = _H // pow(self.pool_kernel, _conv_num)
        _W_linear: int = _W // pow(self.pool_kernel, _conv_num)
        self.features_linear:tuple = mymu.get_cnn_feature_size(_cnn_ch[-1] * _H_linear * _W_linear, _linear_layers, _output_size)

        super(CNN3linear, self).__init__()
        ### Initialize the convolutional layers
        _convs = [nn.Conv2d(_input_size[1], _cnn_ch[0], kernel_size=3, stride=1, padding=1)]
        for idx in range(_conv_num - 1):
            _convs.append(nn.Conv2d(_cnn_ch[idx], _cnn_ch[idx + 1], kernel_size=3, stride=1, padding=1))
        self.convs = nn.ModuleList(_convs)
        ### Initialize the fully connected layers
        self.fc1 = nn.Linear(self.features_linear[0], self.features_linear[1])
        self.fc2 = nn.Linear(self.features_linear[1], self.features_linear[2])
        self.fc3 = nn.Linear(self.features_linear[2], _output_size)
        self.flatten = nn.Flatten()

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. The input is passed through the convolutional layers and then through the fully connected layers.
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: _x: Output tensor of shape (batch_size, 10).
        """
        _x = x
        for conv in self.convs:
            _x = conv(_x)
            _x = nnf.relu(_x)
            _x = nnf.max_pool2d(_x, kernel_size=self.pool_kernel)

        _x = self.flatten(_x)
        _x = self.fc1(_x)
        _x = nnf.relu(_x)
        _x = self.fc2(_x)
        _x = nnf.relu(_x)
        _x = self.fc3(_x)
        return _x


class CNNforCIFAR10(nn.Module):
    def __init__(self, input_size: None | tuple[int, int, int, int] = None, output_size: None | int = None, conv_num: int | None = None) -> None:
        """
        Initialize the model. The model is a simple convolutional neural network model that is designed for CIFAR-10 with 6 convolutional layers and 3 fully connected layers.
        :param input_size: (batch_size, channels, height, width). Input size of the model. Default is None. If None, error occurs.
        :param output_size: Output size of the model. Default is None. If None, the output size is set to 10.
        :param conv_num: Number of convolutional layers. Default is None. If None, the number of convolutional layers is set to 3.
        """
        if input_size is None:
            raise ValueError("param 'input_size' cannot be None")
        else:
            _input_size = input_size

        if output_size is None:
            _output_size = 10
        else:
            _output_size = output_size

        if conv_num is None:
            _conv_num = 3
        else:
            _conv_num = conv_num

        ### Assume that CNN channels double(*2) for each layer, and linear layers are 3 here.
        self.pool_kernel = 2
        _pool_num = 3
        _linear_layers = 2
        _cnn_ch = mymu.get_cnn_channels(16, _conv_num) ## (16, 32, 64, ...) when using cifar10 32 * 32.
        _channels = _input_size[1]
        _H = _input_size[2] ## height. 32 when using cifar10.
        _W = _input_size[3] ## width. 32 when using cifar10.
        _H_linear: int = _H // pow(self.pool_kernel, _pool_num)
        _W_linear: int = _W // pow(self.pool_kernel, _pool_num)
        self.features_linear:tuple = mymu.get_cnn_feature_size(_cnn_ch[-1] * _H_linear * _W_linear, _linear_layers, _output_size)

        super(CNNforCIFAR10, self).__init__()
        ### Initialize the convolutional layers.
        self.conv1 = nn.Conv2d(_input_size[1], _cnn_ch[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(_cnn_ch[0], _cnn_ch[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(_cnn_ch[1], _cnn_ch[2], kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(_cnn_ch[2], _cnn_ch[3], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(_cnn_ch[3], _cnn_ch[4], kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(_cnn_ch[4], _cnn_ch[5], kernel_size=3, stride=1, padding=1)

        ### Initialize Batch Normalization.
        self.bn1 = nn.BatchNorm2d(_cnn_ch[0])
        self.bn2 = nn.BatchNorm2d(_cnn_ch[1])
        self.bn3 = nn.BatchNorm2d(_cnn_ch[2])
        self.bn4 = nn.BatchNorm2d(_cnn_ch[3])
        self.bn5 = nn.BatchNorm2d(_cnn_ch[4])
        self.bn6 = nn.BatchNorm2d(_cnn_ch[5])

        ### Initialize the pooling layers. _pool_num = 3
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool_kernel)
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool_kernel)
        self.pool3 = nn.MaxPool2d(kernel_size=self.pool_kernel)

        ### Initialize the fully connected layers. _linear_layers = 2
        self.fc1 = nn.Linear(self.features_linear[0], self.features_linear[1])
        self.fc2 = nn.Linear(self.features_linear[1], _output_size)
        self.flatten = nn.Flatten()

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. The input is passed through the convolutional layers and then through the fully connected layers.
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: _x: Output tensor of shape (batch_size, 10).
        """
        _x = x
        _x = self.conv1(_x)
        _x = self.bn1(_x)
        _x = nnf.relu(_x)
        _x = self.conv2(_x)
        _x = self.bn2(_x)
        _x = nnf.relu(_x)
        _x = self.pool1(_x)

        _x = self.conv3(_x)
        _x = self.bn3(_x)
        _x = nnf.relu(_x)
        _x = self.conv4(_x)
        _x = self.bn4(_x)
        _x = nnf.relu(_x)
        _x = self.pool2(_x)

        _x = self.conv5(_x)
        _x = self.bn5(_x)
        _x = nnf.relu(_x)
        _x = self.conv6(_x)
        _x = self.bn6(_x)
        _x = nnf.relu(_x)
        _x = self.pool3(_x)

        _x = self.flatten(_x)
        _x = self.fc1(_x)
        _x = nnf.relu(_x)
        _x = self.fc2(_x)

        return _x


class _RoIHead(nn.Module):
    def __init__(self, in_channels:int, roi_size:int, num_classes:int) -> None:
        """
        Initialize the model. The model is a simple Region of Interest (RoI) head for Faster R-CNN.
        :param in_channels: The number of input channels.
        :param roi_size: The size of the RoI. Assume roi_size is square so that H = W.
        :param num_classes: The number of classes. This must include the background class.
        :returns: None
        """
        super().__init__()
        self.in_channels = in_channels
        self.roi_size = roi_size
        self.num_classes = num_classes  ## Total number of classes (including background)
        ### FC layers to spread ROI features (CxHxW) as one vector
        self.fc1 = nn.Linear(in_channels * roi_size * roi_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        ### For classification and bounding box regression
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, roi_features:tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model. The input is passed through the fully connected layers.
        :param roi_features: Input tensor of shape (num_roi, in_channels, roi_size, roi_size).
        :return: scores: Output tensor of shape ((N, num_classes), (N, num_classes*4)).
        """
        _x = nn.Flatten(start_dim=1)(roi_features) ## roi_features: [num_roi, C, H, W]
        _x = self.fc1(_x)
        _x = nnf.relu(_x)
        _x = self.fc2(_x)
        _x = nnf.relu(_x)
        scores = self.cls_score(_x)  ## (N, num_classes)
        bbox_deltas = self.bbox_pred(_x)  ## (N, num_classes*4)
        return scores, bbox_deltas


class MyFasterRCNN(nn.Module):
    def __init__(self, roi_size:int, num_classes:int) -> None:
        """
        Initialize the model. The model is a simple Faster R-CNN model.
        :param roi_size: The size of the RoI. Assume roi_size is square so that H = W.
        :param num_classes: The number of classes. This must include the background class.
        :returns: None
        """
        super().__init__()
        self.backbone = mymu.get_rcnn_backbone()
        self.roi_size = roi_size
        self.num_classes = num_classes
        self.roi_head = _RoIHead(in_channels=2048, roi_size=roi_size, num_classes=num_classes)

    def forward(self, images:tuple[int, int, int, int], proposals:list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model. The input is passed through the backbone and then through the RoI head.
        :param images: Tensor [N, C, H, W] (N = Batch size, C = Channels, H = Height, W = Width)
        :param proposals: List[Tensor[K_i, 4]]
        :return: (cls_scores, bbox_preds): Output tensor of shape ((N, num_classes), (N, num_classes * 4)).
        """
        ### Backbone forward -> Feature map
        feat_maps = self.backbone(images)  ## [N, C, H_feat, W_feat]
        ### RoI Align -> RoI features. Using RoIAlign to extract features from the feature map.
        roi_feats = ops.roi_align(feat_maps, proposals, output_size=(self.roi_size, self.roi_size), spatial_scale=1/32)  ## ResNet50 output stride=32
        ### RoI head forward -> Classification and bounding box regression
        cls_scores, bbox_preds = self.roi_head(roi_feats)
        return cls_scores, bbox_preds