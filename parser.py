
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--upscale_input", type=bool, default=False,
                        help="Upscale imput images to 600x800")
    parser.add_argument("--downscale_input", type=bool, default=False,
                        help="Downscale imput images to 240x320")
    parser.add_argument("--augment_input", type=str, default="None",
                        choices=["None", "grayscale", "color_jitter", "sharpness_adjust"])
    parser.add_argument("--backbone", type=str,
                        default="resnet18", choices=["resnet18", "resnet50", "resnet50moco"])
    parser.add_argument("--use_netvlad", type=bool,
                        help="Specify if NetVLAD must be used")

    parser.add_argument("--netvlad_clusters", type=int, default=64,
                        help="Clusters number for NetVLAD")

    parser.add_argument("--use_gem", type=bool,
                        help="Specify if GeM must be used")

    parser.add_argument("--gem_p", type=int, default=3,
                        help="Power for GeM")

    parser.add_argument("--gem_eps", type=float, default=0.000001,
                        help="Epsilon for GeM")

    parser.add_argument("--use_sgd", type=bool,
                        help="Specify if optimizer sgd must be used")

    parser.add_argument("--use_adagrad", type=bool,
                        help="Specify if optimizer adagrad must be used")

    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Specify momentum for SGD")

    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Maximum number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float,
                        default=0.00001, help="Learning rate")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    # Other parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8,
                        help="num_workers for all dataloaders")
    parser.add_argument("--val_positive_dist_threshold", type=int,
                        default=25, help="Val/test threshold in meters")
    parser.add_argument("--train_positives_dist_threshold",
                        type=int, default=10, help="Train threshold in meters")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str,
                        required=True, help="Path with datasets")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")

    # Test parameters
    parser.add_argument("--test_dataset_name", type=str,
                        help="Name for the dataset to use for evaluation")

    parser.add_argument("--test_model_path", type=str,
                        help="Path for the model to use for evaluation")

    args = parser.parse_args()

    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    return args
