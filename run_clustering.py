
import parser
import datasets_ws
import network
import torch.nn as nn
from clustering import get_clusters

# Parse arguments
args = parser.parse_arguments()

# Creation of Datasets
print(f"Loading train dataset Pitts30k from folder {args.datasets_folder}")
dataset = datasets_ws.BaseDataset(args, args.datasets_folder, "pitts30k", "train")

# Initialize model
model =  nn.Sequential(network.get_backbone(args), network.L2Norm())
model.to(args.device)

print(f"Clustering started...")
get_clusters(args, dataset, model)