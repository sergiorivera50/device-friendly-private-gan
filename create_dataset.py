import argparse
import random

import numpy as np
from torch import optim
from tqdm import tqdm
import os
import torch
import torchvision.utils as vutils
from PIL import Image

from models import Generator, Discriminator
from utils import load_checkpoint


def save_dataset(teacher_model, num_samples, noise_dim, teacher_dir, device):
    """
    Generate images using the teacher model and save them along with the noise vectors.

    :param teacher_model: The pre-trained teacher GAN model.
    :param num_samples: Number of samples to generate.
    :param noise_dim: Dimension of the latent noise vector.
    :param teacher_dir: Base directory to save the dataset.
    :param device: Device on which to perform computations.
    """
    teacher_model.eval()

    # Create directories for inputs and targets
    inputs_dir = os.path.join(teacher_dir, "inputs")
    targets_dir = os.path.join(teacher_dir, "targets")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)

    pbar = tqdm(range(num_samples), desc="Generating Dataset")

    for i in pbar:
        # Generate latent noise
        noise = torch.randn(noise_dim, device=device)

        # Generate image
        with torch.no_grad():
            generated_image = teacher_model(noise).detach().cpu()

        # Save the latent noise vector
        noise_np = noise.cpu().numpy()
        np.save(os.path.join(inputs_dir, f"{i}.npy"), noise_np)

        # Save the generated image
        img = vutils.make_grid(generated_image, normalize=True)
        img_np = np.transpose(img.numpy(), (1, 2, 0))
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil.save(os.path.join(targets_dir, f"{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="manual seed for reproducibility"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="number of samples for the teacher network to generate"
    )
    parser.add_argument(
        "--teacher-dir",
        type=str,
        default="./out/checkpoints/DCGAN_30e/ckpt_29e.pth",
        help="base directory of the checkpoint file for the teacher network"
    )
    args = parser.parse_args()
    print(args)

    nz = 100
    generator = Generator(nz=nz, ngf=64, nc=3, quant=False)
    discriminator = Discriminator(ndf=64, nc=3)
    load_checkpoint(
        path=args.teacher_dir,
        generator=generator,
        discriminator=discriminator,
        generator_optim=optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
        discriminator_optim=optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
    )

    # Set seed for reproducibility

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    save_dataset(
        teacher_model=generator,
        num_samples=args.num_samples,
        noise_dim=(1, nz, 1, 1),
        teacher_dir="./data/teacher",
        device="cpu"
    )
