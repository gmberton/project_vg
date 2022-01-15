import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from netvlad import NetVLAD
from gem import GeM
import h5py
from os.path import join


class GeoLocalizationNet(nn.Module):
    """The model is composed of a backbone and an aggregation layer.
    The backbone is a (cropped) ResNet-18, and the aggregation is a L2
    normalization followed by max pooling.
    """

    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)

        if args.use_gem:
            self.aggregation = nn.Sequential(
                GeM(p=args.gem_p, eps=args.gem_eps), Flatten(), L2Norm())

        elif args.use_netvlad:
            initcache = join(args.datasets_folder, 'centroids_' + str(
                args.netvlad_clusters) + '_' + str(args.backbone) + '_desc_cen.hdf5')
            self.aggregation = NetVLAD(
                num_clusters=args.netvlad_clusters, dim=args.features_dim)
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.aggregation.init_params(clsts, traindescs)
                del clsts, traindescs

            # Number of output features from NetVLAD
            args.features_dim *= args.netvlad_clusters
        else:
            self.aggregation = nn.Sequential(L2Norm(),
                                             torch.nn.AdaptiveAvgPool2d(1),
                                             Flatten())

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_backbone(args):

    features_dim = 0
    if args.backbone == 'resnet18':
        features_dim = 256
        backbone = torchvision.models.resnet18(pretrained=True)
        for name, child in backbone.named_children():
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            "Train only conv4 of the ResNet-18 (remove conv5), freeze the previous ones")
        layers = list(backbone.children())[:-3]
        backbone = torch.nn.Sequential(*layers)

    elif args.backbone == 'resnet50':
        features_dim = 2048
        backbone = torchvision.models.resnet50(pretrained=True)
        for name, child in backbone.named_children():
            if name == "layer4":
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            "Train only conv5 of the ResNet-50, freeze the previous ones")
        layers = list(backbone.children())[:-2]
        backbone = torch.nn.Sequential(*layers)

    elif args.backbone == 'resnet50moco':
        features_dim = 2048
        backbone = torch.load('moco_v1_200ep_pretrain.pth.tar')
        for name, child in backbone.named_children():
            if name == "layer4":
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            "Train only conv5 of the ResNet-50 trained by MoCo-v1 team, freeze the previous ones")
        layers = list(backbone.children())[:-2]
        backbone = torch.nn.Sequential(*layers)

    args.features_dim = features_dim  # Number of output features from backbone
    return backbone


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
