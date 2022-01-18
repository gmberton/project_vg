
import datasets_ws
import network
import commons
import parser
import test
import torch
import logging

import multiprocessing
from os.path import join
from datetime import datetime
torch.backends.cudnn.benchmark = True  # Provides a speedup

if __name__ == "__main__":
    # Initial setup: parser, logging...
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.output_folder = join("test_runs", args.exp_name,
                              start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.output_folder)
    commons.make_deterministic(args.seed)

    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")
    logging.info(
        f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    # Creation of Datasets
    logging.debug(
        f"Loading test dataset {args.test_dataset_name} from folder {args.datasets_folder}")

    test_ds = datasets_ws.BaseDataset(
        args, args.datasets_folder, args.test_dataset_name, "test")
    logging.info(f"Test set: {test_ds}")

    # Initialize model
    model = network.GeoLocalizationNet(args)
    model = model.to(args.device)

    # Test best model on test set
    model_state_dict = torch.load(
        args.test_model_path, map_location=torch.device(args.device))["model_state_dict"]
    model.load_state_dict(model_state_dict)

    recalls, recalls_str = test.test(args, test_ds, model)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")
