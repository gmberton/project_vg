
import torch
import shutil
from os.path import join

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.output_folder, "best_model.pth"))

