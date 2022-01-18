
import torch
import shutil
from os.path import join
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.output_folder, "best_model.pth"))


def save_attention_mask(images, masks):
    W = images[0].shape[2]
    H = images[0].shape[1]

    # Create figure and array of axes
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=[10, 10])

    for i, axi in enumerate(ax.flat):
        image = images[i].permute(1, 2, 0)
        mask = masks[i].reshape(30, 40)

        filtered_mask = gaussian_filter(mask, 3)

        axi.imshow(image[:, :, 0], cmap='gray')
        axi.imshow(filtered_mask[:, :], cmap='rainbow', interpolation='bilinear',
                   alpha=0.8, extent=(0, 640, 480, 0))

    plt.show()
