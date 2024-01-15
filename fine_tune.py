import argparse
import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from opacus import PrivacyEngine
from torch import optim, nn
from tqdm import tqdm

from models import Generator, Discriminator
from utils import create_dir, save_checkpoint, compute_perceptual_loss, soft_threshold, load_checkpoint
from utils.data import TeacherGeneratedDataset
from utils.vgg import CustomVgg16

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
        default=1e-5,
        help="learning rate for the Adam optimizer"
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-3,
        help="weight decay"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=20,
        help="weight for the GAN loss"
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
    dataset = TeacherGeneratedDataset(
        root="./data/teacher",
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    # Create dataloader

    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                 drop_last=True)

    # Build and train GAN model
    nc = 3
    nz = 100
    ckpt = torch.load("./out/compressed16b.pth")
    generator = Generator(nz=nz, ngf=64, nc=nc, dims=ckpt["channels"], quant=True).to(device)
    generator.load_state_dict(ckpt["weights"])
    discriminator = Discriminator(ndf=64, nc=nc).to(device)
    discriminator.apply(weights_init)

    beta1 = 0.5
    criterion = nn.BCELoss()

    # Optimisers
    generator_optim = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.wd, betas=(beta1, 0.999))
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.wd, betas=(beta1, 0.999))

    # Learning Rate Schedulers
    generator_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(generator_optim, args.epochs)
    discriminator_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(discriminator_optim, args.epochs)

    # Setup differential privacy for the discriminator module
    delta = 1e-5
    privacy_engine = PrivacyEngine(secure_mode=True)
    discriminator, discriminator_optim, dataloader = privacy_engine.make_private(
        module=discriminator,
        optimizer=discriminator_optim,
        data_loader=dataloader,
        noise_multiplier=1.0,
        max_grad_norm=1e-4,
        batch_first=True,
    )

    input_shape = (args.batch_size, nz, 1, 1)
    fixed_noise = torch.randn(input_shape, device=device)

    g_losses, g_losses_perceptual, g_losses_GAN, d_losses = [], [], [], []

    iters = 0
    img_progress = []
    channel_number = []

    vgg = CustomVgg16()  # to extract features of student vs teacher
    torch.autograd.set_detect_anomaly(True)

    # For each epoch
    for epoch in range(args.epochs):
        generator.train(), discriminator.train()
        pbar = tqdm(dataloader)
        # For each batch in the dataloader
        for i, data in enumerate(pbar, 0):
            # print("Input shape:", data["input"].shape)
            # print("Input source shape:", input_source.shape)
            # print("Target shape:", data["target"].shape)
            # print("Target source shape:", target_source.shape)

            input_data = data["input"].to(device)
            batch_size = input_data.size(0)

            # Take model input and target image
            # print("batch_size:", batch_size)
            input_vector = data["input"].view(batch_size, nz, 1, 1).to(device)  # z
            target_img = data["target"].to(device)  # G_t(z)

            # 1. Update G network (student)
            generator_optim.zero_grad()

            student_output = generator(input_vector)

            # Compute perceptual loss (to evaluate difference with teacher)

            student_output_features = vgg(student_output.cpu())
            teacher_output_features = vgg(target_img.cpu())
            g_loss_perceptual, g_loss_content, g_loss_style = compute_perceptual_loss(
                student_output_features, teacher_output_features
            )

            # Compute GAN loss (for G)
            pred_student_output = discriminator(student_output)
            real_labels = torch.full((batch_size,), REAL_LABEL, dtype=torch.float, device=device, requires_grad=False)
            g_loss_GAN = criterion(pred_student_output, real_labels)

            # Compute total G network loss (student)
            g_loss = args.beta * g_loss_perceptual + g_loss_GAN
            g_loss.backward()

            generator_optim.step()

            g_losses.append(g_loss.item())
            g_losses_perceptual.append(g_loss_perceptual.item())
            g_losses_GAN.append(g_loss_GAN.item())

            # 2. Update D network
            discriminator_optim.zero_grad()

            # Compute real loss
            pred_teacher_output = discriminator(target_img)
            d_loss_real = criterion(pred_teacher_output, real_labels)
            expected_output_real = pred_teacher_output.mean().item()

            # Compute fake loss
            student_output = generator(input_vector)
            pred_student_output = discriminator(student_output)
            fake_labels = torch.full((batch_size,), FAKE_LABEL, dtype=torch.float, device=device, requires_grad=False)
            d_loss_fake = criterion(pred_student_output, fake_labels)
            expected_output_fake = pred_student_output.mean().item()

            # Compute total loss
            d_loss = (d_loss_real + d_loss_fake)
            d_loss.backward()

            discriminator_optim.step()
            d_losses.append(d_loss.item())

            epsilon = privacy_engine.accountant.get_epsilon(delta=delta)
            pbar.set_description(
                f"[Epoch {epoch + 1}/{args.epochs}] Loss_D: {d_loss.item():.4f} "
                f"Loss_G: {g_loss.item():.4f} D(x): {expected_output_real:.4f} "
                f"Loss_G_Perceptual: {g_loss_perceptual:.4f} Loss_G_GAN: {g_loss_GAN:.4f} "
                f"D(G(z)): {expected_output_fake:.4f} "
                f"(ε = {epsilon:.2f}, δ = {delta:.2f})"
            )

            # 5. Track generator progress by saving G's output on fixed noise
            if (iters % 500 == 0) or ((epoch == args.epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_progress.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1

        # At the end of each epoch
        generator.eval(), discriminator.eval()

        # Update learning rates
        generator_lr_scheduler.step()
        discriminator_lr_scheduler.step()

        # Save current model checkpoint
        model_path = os.path.join("out", "checkpoints", "dp_" + str(args.epochs) + "e16b")
        create_dir(model_path)
        save_checkpoint(
            path=os.path.join(model_path, "ckpt_" + str(epoch) + "e.pth"),
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            generator_optim=generator_optim,
            discriminator_optim=discriminator_optim,
            generator_lr_scheduler=generator_lr_scheduler,
            discriminator_lr_scheduler=discriminator_lr_scheduler,
            g_losses=g_losses,
            g_losses_perceptual=g_losses_perceptual,
            g_losses_GAN=g_losses_GAN,
            d_losses=d_losses,
            fixed_noise=fixed_noise,
            img_progress=img_progress,
            channel_number=channel_number,
        )
    print("Finished Training")
