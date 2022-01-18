
from __future__ import print_function

from math import ceil
import random, shutil, json
from os.path import join, exists
from os import makedirs

import network 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import h5py
import faiss
import numpy as np
import netvlad
import parser
import datasets_ws

args = parser.parse_arguments()
nDescriptors = 50000
nPerImage = 100
nIm = ceil(nDescriptors/nPerImage)

encoder_dim = 256

#create a baseDataset object for the given training ds, used to compute centroids 
cluster_set = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "train")

model = network.get_backbone(args)
model.to(args.device)

sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
data_loader = DataLoader(dataset=cluster_set, 
            num_workers=args.num_workers, batch_size=args.infer_batch_size, shuffle=False, 
            sampler=sampler)

if not exists(join(args.dataset_folder, 'centroids')):
    makedirs(join(args.dataset_folder, 'centroids'))


initcache = join(args.datasets_folder, 'centroids', args.dataset_name + '_' + str(args.num_clusters) + '_desc_cen.hdf5')
with h5py.File(initcache, mode='w') as h5: 
    with torch.no_grad():
        model.eval()
        print('====> Extracting Descriptors')
        dbFeat = h5.create_dataset("descriptors", 
                    [nDescriptors, encoder_dim], 
                    dtype=np.float32)

        for iteration, (input, indices) in enumerate(data_loader, 1):
            input = input.to(args.device)
            image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

            batchix = (iteration-1)*args.infer_batch_size*nPerImage
            for ix in range(image_descriptors.size(0)):
                # sample different location for each image in batch
                sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                startix = batchix + ix*nPerImage
                dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

            if iteration % 50 == 0 or len(data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    ceil(nIm/args.infer_batch_size)), flush=True)
            del input, image_descriptors
    
    print('====> Clustering..')
    niter = 100
    kmeans = faiss.Kmeans(encoder_dim, args.num_clusters, niter=niter, verbose=False)
    kmeans.train(dbFeat[...])

    print('====> Storing centroids', kmeans.centroids.shape)
    h5.create_dataset('centroids', data=kmeans.centroids)
    print('====> Done!')