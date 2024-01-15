import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import optim, nn
from torch.autograd import Variable
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
        "--lr-gamma",
        type=float,
        default=1e-1,
        help="learning rate for channel pruning (gamma)"
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
        "--rho",
        type=float,
        default=0.01,
        help="weight for the L1 loss"
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
    generator = Generator(nz=nz, ngf=64, nc=nc, quant=True).to(device)
    generator.apply(weights_init)
    discriminator = Discriminator(ndf=64, nc=nc).to(device)
    discriminator.apply(weights_init)

    beta1 = 0.5
    criterion = nn.BCELoss()

    # Optimisers
    generator_optim = optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.wd, betas=(beta1, 0.999))
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.wd, betas=(beta1, 0.999))
    parameters_gamma = []
    for name, param in generator.named_parameters():
        if 'weight' in name and param.ndimension() == 1:
            parameters_gamma.append(param)
    print('parameters_gamma:', len(parameters_gamma))
    gamma_optim = optim.SGD(parameters_gamma, lr=args.lr_gamma, momentum=0.5)

    # Learning Rate Schedulers
    generator_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(generator_optim, args.epochs)
    discriminator_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(discriminator_optim, args.epochs)
    gamma_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(gamma_optim, args.epochs)

    # Memory allocation
    Tensor = torch.Tensor
    input_source = Tensor(args.batch_size, nz, 1, 1).to(device)  # latent vector
    target_source = Tensor(args.batch_size, nc, image_size, image_size).to(device)  # teacher-generated image

    input_shape = (args.batch_size, nz, 1, 1)
    fixed_noise = torch.randn(input_shape, device=device)

    g_losses, g_losses_perceptual, g_losses_GAN, d_losses = [], [], [], []

    iters = 0
    img_progress = []
    channel_number = []

    vgg = CustomVgg16()  # to extract features of student vs teacher
    torch.autograd.set_detect_anomaly(True)

    _ = load_checkpoint(
        path="./out/checkpoints/DCGAN_30e/ckpt_29e.pth",
        generator=generator,
    )

    scaler_list = []
    for m in generator.modules():
        if isinstance(m, torch.nn.GroupNorm) and m.weight is not None:
            m_cpu = m.weight.data.cpu().numpy().squeeze()
            # print('m_cpu:', type(m_cpu), m_cpu.shape)
            scaler_list.append(m_cpu)
    all_scaler = np.concatenate(scaler_list, axis=0)
    l0norm = np.sum(all_scaler != 0)
    channel_number.append(l0norm)
    print(f"Initial non-zero channels: {l0norm}")

    print(f"Starting training for {args.epochs} epochs")
    # For each epoch
    for epoch in range(args.epochs):
        generator.train(), discriminator.train()
        pbar = tqdm(dataloader)
        real_labels = torch.full((args.batch_size,), REAL_LABEL, dtype=torch.float, device=device, requires_grad=False)
        fake_labels = torch.full((args.batch_size,), FAKE_LABEL, dtype=torch.float, device=device, requires_grad=False)
        # For each batch in the dataloader
        for i, data in enumerate(pbar, 0):
            # print("Input shape:", data["input"].shape)
            # print("Input source shape:", input_source.shape)
            # print("Target shape:", data["target"].shape)
            # print("Target source shape:", target_source.shape)

            input_data = data["input"].to(device)
            batch_size = input_data.size(0)

            # Take model input and target image
            input_vector = Variable(input_source.copy_(data["input"].view(batch_size, nz, 1, 1)))  # z
            target_img = Variable(target_source.copy_(data["target"]))  # G_t(z)

            # 1. Update G network (student)
            generator_optim.zero_grad()
            gamma_optim.zero_grad()

            student_output = generator(input_vector)

            # Compute perceptual loss (to evaluate difference with teacher)

            student_output_features = vgg(student_output.cpu())
            teacher_output_features = vgg(target_img.cpu())
            g_loss_perceptual, g_loss_content, g_loss_style = compute_perceptual_loss(
                student_output_features, teacher_output_features
            )

            # Compute GAN loss (for G)
            pred_student_output = discriminator(student_output)
            g_loss_GAN = criterion(pred_student_output, real_labels)

            # Compute total G network loss (student)
            g_loss = args.beta * g_loss_perceptual + g_loss_GAN
            g_loss.backward()

            generator_optim.step()
            gamma_optim.step()

            g_losses.append(g_loss.item())
            g_losses_perceptual.append(g_loss_perceptual.item())
            g_losses_GAN.append(g_loss_GAN.item())

            # Proximal gradient (channel pruning)
            gamma_lr = gamma_lr_scheduler.get_last_lr()[0]
            p = float(gamma_lr) * float(args.rho)
            for name, m in generator.named_modules():
                if isinstance(m, nn.GroupNorm) and m.weight is not None:
                    m.weight.data = soft_threshold(m.weight.data, th=p)

            # 2. Update D network
            discriminator_optim.zero_grad()

            # Compute real loss
            pred_teacher_output = discriminator(target_img)
            d_loss_real = criterion(pred_teacher_output, real_labels)
            expected_output_real = pred_teacher_output.mean().item()

            # Compute fake loss
            student_output = generator(input_vector)
            pred_student_output = discriminator(student_output)
            d_loss_fake = criterion(pred_student_output, fake_labels)
            expected_output_fake = pred_student_output.mean().item()

            # Compute total loss
            d_loss = (d_loss_real + d_loss_fake)
            d_loss.backward()

            discriminator_optim.step()
            d_losses.append(d_loss.item())

            pbar.set_description(
                f"[Epoch {epoch + 1}/{args.epochs}] Loss_D: {d_loss.item():.4f} "
                f"Loss_G: {g_loss.item():.4f} D(x): {expected_output_real:.4f} "
                f"Loss_G_Perceptual: {g_loss_perceptual:.4f} Loss_G_GAN: {g_loss_GAN:.4f} "
                f"D(G(z)): {expected_output_fake:.4f} gamma_lr: {gamma_lr:.4f} p: {p:.4f}"
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
        gamma_lr_scheduler.step()

        # Store number of non-zero channels
        channels = []
        for m in generator.modules():
            if isinstance(m, torch.nn.GroupNorm) and m.weight is not None:
                m_cpu = m.weight.data.cpu().numpy().squeeze()
                channels.append(m_cpu)
        channels = np.concatenate(channels, axis=0)
        l0norm = np.sum(channels != 0)
        channel_number.append(l0norm)
        print(f"Non-zero channels: {l0norm}")

        # Save current model checkpoint
        model_path = os.path.join("out", "checkpoints", "student_" + str(args.epochs) + "e8b")
        create_dir(model_path)
        save_checkpoint(
            path=os.path.join(model_path, "ckpt_" + str(epoch) + "e.pth"),
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            generator_optim=generator_optim,
            gamma_optim=gamma_optim,
            discriminator_optim=discriminator_optim,
            generator_lr_scheduler=generator_lr_scheduler,
            discriminator_lr_scheduler=discriminator_lr_scheduler,
            gamma_lr_scheduler=gamma_lr_scheduler,
            g_losses=g_losses,
            g_losses_perceptual=g_losses_perceptual,
            g_losses_GAN=g_losses_GAN,
            d_losses=d_losses,
            fixed_noise=fixed_noise,
            img_progress=img_progress,
            channel_number=channel_number,
        )
    print("Finished Training")
