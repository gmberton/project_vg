
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


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def save_attention_mask(images, masks):
    W = images[0].shape[2]
    H = images[0].shape[1]

    mask_W = W//16
    mask_H = H//16

    # Create figure and array of axes
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=[10, 10])

    for i, axi in enumerate(ax.flat):
        image = inverse_normalize(tensor=images[i], mean=(
            0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).permute(1, 2, 0)
        mask = masks[i].reshape(mask_H, mask_W)

        filtered_mask = gaussian_filter(mask, 3)

        axi.imshow(image[:, :, :])
        axi.imshow(filtered_mask[:, :], cmap='turbo', interpolation='bilinear',
                   alpha=0.6, extent=(0, W, H, 0))

    plt.show()
