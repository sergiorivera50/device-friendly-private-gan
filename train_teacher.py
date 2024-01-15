import argparse
import os
import random
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import optim, nn
from tqdm import tqdm

from models import Generator, Discriminator
from utils import create_dir, save_checkpoint

NUM_WORKERS = 0  # when > 0, increased risk of shared memory manager timing out

REAL_LABEL = 1
FAKE_LABEL = 0


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="manual seed for reproducibility"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="number of epochs to normalize the training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="learning rate for the Adam optimizer"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="batch size for training"
    )
    args = parser.parse_args()
    print(args)

    # Set seed for reproducibility

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # Select device ("mps" for Apple M1, else "cpu")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load CelebA dataset

    image_size = 64
    dataset = datasets.ImageFolder(
        root="./data/celeba",
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    print("dataset:", len(dataset))

    # Create dataloader

    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    # Build and train GAN model

    nz = 100
    generator = Generator(nz=nz, ngf=64, nc=3, quant=False).to(device)
    generator.apply(weights_init)
    discriminator = Discriminator(ndf=64, nc=3).to(device)
    discriminator.apply(weights_init)

    beta1 = 0.5
    criterion = nn.BCELoss()
    generator_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(beta1, 0.999))
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(beta1, 0.999))

    input_shape = (64, nz, 1, 1)
    fixed_noise = torch.randn(input_shape, device=device)

    g_losses, d_losses = [], []

    iters = 0
    img_progress = []

    print(f"Starting training for {args.epochs} epochs")
    # For each epoch
    for epoch in range(args.epochs):
        generator.train(), discriminator.train()
        pbar = tqdm(dataloader)
        # For each batch in the dataloader
        for i, data in enumerate(pbar, 0):
            # 1. Update D network: maximize log(D(x)) + log(1 - D(G(z)))

            # 1.1. Train with all-real batch
            discriminator.zero_grad(set_to_none=True)
            # Format batch
            real_data = data[0].to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_data)
            # Calculate loss on the all-real batch
            discriminator_error_real = criterion(output, label)
            # Compute D(x)
            expected_output_real = output.mean().item()

            # 1.2. Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label = torch.fill(label, FAKE_LABEL)
            # Classify all fake batch with D
            output = discriminator(fake.detach())
            # Calculate D's loss on the all-fake batch
            discriminator_error_fake = criterion(output, label)
            # Compute D(G(z_1))
            expected_output_fake_before = output.mean().item()

            # Update D - error as sum over the fake and the real batches
            discriminator_error = discriminator_error_real + discriminator_error_fake
            discriminator_error.backward()
            discriminator_optim.step()
            discriminator_optim.zero_grad(set_to_none=True)

            # 2. Update G network: maximize log(D(G(z)))
            generator.zero_grad()
            label = torch.fill(label, REAL_LABEL)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake)
            # Update G - loss based on this output
            generator_error = criterion(output, label)
            generator_error.backward()
            generator_optim.step()
            # Compute D(G(z_2))
            expected_output_fake_after = output.mean().item()
            pbar.set_description(
                f"[Epoch {epoch+1}/{args.epochs}] Loss_D: {discriminator_error.item():.4f} "
                f"Loss_G: {generator_error.item():.4f} D(x): {expected_output_real:.4f} "
                f"D(G(z)): {expected_output_fake_before:.4f} / {expected_output_fake_after:.4f} "
            )

            # 4. Save losses for plotting later
            g_losses.append(generator_error.item())
            d_losses.append(discriminator_error.item())

            # 5. Track generator progress by saving G's output on fixed noise
            if (iters % 500 == 0) or ((epoch == args.epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_progress.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1

        # At the end of each epoch, save current model checkpoint
        model_path = os.path.join("out", "checkpoints", "DCGAN_" + str(args.epochs) + "e")
        create_dir(model_path)
        save_checkpoint(
            path=os.path.join(model_path, "ckpt_" + str(epoch) + "e.pth"),
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            generator_optim=generator_optim,
            discriminator_optim=discriminator_optim,
            g_losses=g_losses,
            d_losses=d_losses,
            fixed_noise=fixed_noise,
            img_progress=img_progress,
        )
    print("Finished Training")
