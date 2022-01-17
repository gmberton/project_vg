import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from crn import CRN
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

        if args.use_attention == "crn":
            self.attention = CRN(args)

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

        reweight_mask = None
        if self.attention:
            reweight_mask = self.attention(x)

        x = self.aggregation(x, reweight_mask)
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

    elif args.backbone == 'alexnet':
        features_dim = 256
        backbone = torchvision.models.alexnet(pretrained=True)

        # capture only features and remove last relu and maxpool
        layers = list(backbone.features.children())[:-2]

        # only train conv5
        for l in layers[:-1]:
            for p in l.parameters():
                p.requires_grad = False
        logging.debug(
            "Train only conv5 of the AlexNet, freeze the previous ones")

        backbone = torch.nn.Sequential(*layers)

    elif args.backbone == 'vgg16':
        features_dim = 512
        backbone = torchvision.models.vgg16(pretrained=True)

        # capture only features and remove last relu and maxpool
        layers = list(backbone.features.children())[:-2]

        #  only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]:
            for p in l.parameters():
                p.requires_grad = False
        logging.debug(
            "Train only conv5_1, conv5_2, conv5_3 of the VGG16, freeze the previous ones")

        backbone = torch.nn.Sequential(*layers)

    elif args.backbone == 'resnet50-conv5':
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

    elif args.backbone == 'resnet50-conv4':
        features_dim = 1024
        backbone = torchvision.models.resnet50(pretrained=True)
        for name, child in backbone.named_children():
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            "Cut conv5, Train only conv4 of the ResNet-50, freeze the previous ones")
        layers = list(backbone.children())[:-3]
        backbone = torch.nn.Sequential(*layers)

    elif args.backbone == 'resnet50moco-conv5':
        features_dim = 2048
        backbone = torch.load('moco_v1_200ep_pretrain.pth.tar')
        for name, child in backbone.named_children():
            if name == "layer4":
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            "Train only conv5 of the ResNet-50 pre-trained by MoCo-v1 team, freeze the previous ones")
        layers = list(backbone.children())[:-2]
        backbone = torch.nn.Sequential(*layers)

    elif args.backbone == 'resnet50moco-conv4':
        features_dim = 1024
        backbone = torch.load('moco_v1_200ep_pretrain.pth.tar')
        for name, child in backbone.named_children():
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(
            "Cut cov5, Train only conv4 of the ResNet-50 pre-trained by MoCo-v1 team, freeze the previous ones")
        layers = list(backbone.children())[:-3]
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
