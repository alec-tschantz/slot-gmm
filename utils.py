import torch

from matplotlib import pyplot as plt


renormalize = lambda x: (x + 1) / 2


def test(model, batch):
    model.eval()

    with torch.no_grad():
        image = batch["image"][0].unsqueeze(0)
        recon_combined, recons, masks, slots = model(image)

        fig, ax = plt.subplots(1, 6, figsize=(10, 2))

        recons = recons.squeeze(0)
        masks = masks.squeeze(0)

        image = image.squeeze(0)
        image = image.permute(1, 2, 0).numpy()
        image = renormalize(image)

        recon_combined = recon_combined.squeeze(0)
        recon_combined = recon_combined.permute(1, 2, 0)
        recon_combined = recon_combined.detach().numpy()
        recon_combined = renormalize(recon_combined)

        ax[0].imshow(image)
        ax[1].imshow(recon_combined)
        for i in range(4):
            slot_image = recons[i] * masks[i] + (1 - masks[i])
            slot_image = slot_image.permute(1, 2, 0).detach().numpy()
            slot_image = renormalize(slot_image)
            ax[i + 2].imshow(slot_image)

        for i in range(len(ax)):
            ax[i].grid(False)
            ax[i].axis("off")

        plt.show()
