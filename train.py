
import math
from tabnanny import check
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
from os.path import exists
import h5py
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import parser
import commons
import network
import datasets_ws

from google.colab import drive 
import os



#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()

# If the user doesn't insert a colab folder, I keep the default folder for output
# otherwise i change it inside the if condition
args.output_folder = join("runs", args.exp_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
colab = False

if args.colab_folder is not None:
    colab = True
    logging.debug("Read folder" + args.colab_folder + "inside your drive")
    args.output_folder = 'drive/MyDrive/'+ args.colab_folder

    # now i can check if the output folder exist 
    # in this case I resume the old train
    if os.path.isdir(args.output_folder):
        #check if my file exists
        files = os.listdir(args.output_folder)
        for f in files:
            if f == "last_model.pth":
                # found a saved best model
                # put a variable to True and load it later
                found_last_model = True

    else:
        # otherwise I create the folder and begin a new train
        os.mkdir(args.output_folder)
  


commons.setup_logging(args.output_folder, resume=colab)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset Pitts30k from folder {args.datasets_folder}")

triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, "pitts30k", "train", args.negs_num_per_query)
logging.info(f"Train query set: {triplets_ds}")

val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, "pitts30k", "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, "pitts30k", "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
model = network.GeoLocalizationNet(args)
model = model.to(args.device)

#### Setup Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

best_r5 = 0
not_improved_num = 0

logging.info(f"Output dimension of the model is {args.features_dim}")

# Take clusters for NetVLAD
if args.type == 'NETVLAD':
    net_vlad = model.aggregation[1]
    initcache = join(args.datasets_folder, 'centroids', args.dataset_name + '_' + str(args.num_clusters) + '_desc_cen.hdf5')

    if not exists(initcache):
        raise FileNotFoundError('Could not find clusters, please run cluster.py before proceeding')

    with h5py.File(initcache, mode='r') as h5: 
        clsts = h5.get("centroids")[...]
        traindescs = h5.get("descriptors")[...]
        net_vlad.init_params(clsts, traindescs) 
        del clsts, traindescs

    model = model.to(args.device)



# If I found a last model, I load it
if found_last_model is True:
    checkpoint = torch.load(args.output_folder + "/last_model.pth")
    epoch_num = checkpoint["epoch_num"] + 1 # start from the next epoch
    model.load_state_dict(checkpoint["model_state_dict"]) # pytorch function
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # pytorch function
    recalls = checkpoint["recalls"]
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]


#### Training loop
while epoch_num < args.epochs_num: 
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")
        
        # Compute triplets to use in the triplet loss
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False
        
        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device=="cuda"),
                                 drop_last=True)
        
        model = model.train()
        
        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
        # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):
            
            # Compute features of all images (images contains queries, positives and negatives)
            features = model(images.to(args.device))
            loss_triplet = 0
            
            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
            for triplets in triplets_local_indexes:
                queries_indexes, positives_indexes, negatives_indexes = triplets.T
                loss_triplet += criterion_triplet(features[queries_indexes],
                                                  features[positives_indexes],
                                                  features[negatives_indexes])
            del features
            loss_triplet /= (args.train_batch_size * args.negs_num_per_query)
            
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()
            
            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet
        
        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                      f"current batch triplet loss = {batch_loss:.4f}, " +
                      f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
    
    is_best = recalls[1] > best_r5
    epoch_num += 1
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_model.pth")
    
    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
        best_r5 = recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break


logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.output_folder, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

