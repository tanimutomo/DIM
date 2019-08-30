import torch
import torch.nn as nn
from collections import OrderedDict


feature_types = ['conv', 'fc', 'glob']
network_types = ['encoder', 'classifier']

class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Convnet('conv_encoder')
        self.classifier = Convnet('conv_classifier')

    def forward(self, x):
        encoding = self.encoder(x)
        out = self.classifier(encoding)
        return out

class Convnet(nn.Module):
    def __init__(self, network):
        super().__init__()

        feature_type, network_type = network.split('_')
        assert feature_type in feature_types
        assert network_type in network_types

        if network_type == 'encoder':
            self.layers = self.make_encoder(feature_type)
        else:
            self.layers = self.make_classifier(feature_type)

    def make_encoder(self, feature_type):
        modules = nn.Sequential()
        modules.add_module('layer0', self.make_conv_layer(3, 64))
        modules.add_module('layer1', self.make_conv_layer(64, 128))
        modules.add_module('layer2', self.make_conv_layer(128, 256))
        if feature_type == 'conv':
            return modules
        modules.add_module('layer3', View(-1, 4096))
        modules.add_module('layer4', self.make_fc_layer(4096, 1024))
        if feature_type == 'fc':
            return modules
        modules.add_module('layer5', nn.Linear(1024, 64, bias=True))
        return modules

    def make_classifier(self, feature_type):
        modules = nn.Sequential()
        if feature_type == 'conv':
            args = dict(in_f=4096, out_f=200, bias=True, 
                        flatten=True, drop=0.1)
        elif feature_type == 'fc':
            args = dict(in_f=1024, out_f=200, bias=True, drop=0.1)
        elif feature_type == 'glob':
            args = dict(in_f=64, out_f=200, bias=True, drop=0.1)

        modules.add_module('layer0', self.make_fc_layer(**args))
        modules.add_module('layer1', nn.Linear(200, 10, bias=True))
        return modules

    def make_conv_layer(self, in_c, out_c, kernel_size=(4, 4), stride=(2, 2),
                        padding=(1, 1), bias=False, eps=1e-05, momentum=0.1,
                        affine=True, track_running_stats=True, inplace=True):
        conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        bn = nn.BatchNorm2d(out_c, eps=eps, momentum=momentum,
                            affine=affine, 
                            track_running_stats=track_running_stats)
        relu = nn.ReLU(inplace=inplace)
        return nn.Sequential(OrderedDict([('conv', conv),
                                          ('bn', bn),
                                          ('ReLU', relu)]))

    def make_fc_layer(self, in_f, out_f, bias=False, eps=1e-05, momentum=0.1,
                      affine=True, track_running_stats=True, inplace=True,
                      flatten=False, drop=False):
        modules = nn.Sequential()
        if flatten:
            modules.add_module('flatten', View(-1, in_f))
        modules.add_module('fc', nn.Linear(in_f, out_f, bias=bias))
        if drop:
            modules.add_module('do', nn.Dropout(p=drop, inplace=False))
        modules.add_module('bn', nn.BatchNorm1d(
                    out_f, eps=eps, 
                    momentum=momentum, affine=affine,
                    track_running_stats=track_running_stats
                    ))
        modules.add_module('ReLU', nn.ReLU(inplace=inplace))
        return modules

    def forward(self, x):
        out = self.layers(x)
        return out


class View(torch.nn.Module):
    """Basic reshape module.
    """
    def __init__(self, *shape):
        """
        Args:
            *shape: Input shape.
        """
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """Reshapes tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: Flattened tensor.
        """
        return input.view(*self.shape)


if __name__ == '__main__':
    model = ConvClassifier()
    model.encoder.load_state_dict(
            torch.load('./models/dim_local_dv_encoder.pth'),
            strict=False
            )
    model.classifier.load_state_dict(
            torch.load('./models/dim_local_dv_conv_classifier.pth'),
            strict=False
            )
    model.eval()
    print(model)
    # encoder = encoder.encoder
    # print(encoder)

    # import torch.optim as optim
    # classifier = 'conv'
    # encoder = BigEncoder(classifier)
    # # encoder.load_state_dict(torch.load('./models/dim_local_dv_encoder.pth'))
    # optimizer = optim.Adam(encoder.parameters())
    # optimizer.zero_grad()
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = model(x)
    print(out.shape)
    # criterion = nn.L1Loss()
    # t = torch.ones(out.shape)
    # loss = criterion(out, t)
    # loss.backward()
    # optimizer.step()
    # torch.save(encoder.state_dict(), 'models/tmp.pth')


