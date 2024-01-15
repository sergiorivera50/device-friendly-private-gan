import os
import torch
import torch.nn as nn
from torch.nn import L1Loss


def save_checkpoint(
        path,
        epoch,
        generator,
        discriminator,
        generator_optim,
        discriminator_optim,
        g_losses,
        d_losses,
        fixed_noise,
        img_progress,
        gamma_optim=None,
        generator_lr_scheduler=None,
        discriminator_lr_scheduler=None,
        gamma_lr_scheduler=None,
        g_losses_perceptual=None,
        g_losses_GAN=None,
        channel_number=None,
):
    ckpt = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_optim": generator_optim.state_dict(),
        "discriminator_optim": discriminator_optim.state_dict(),
        "g_losses": g_losses,
        "d_losses": d_losses,
        "fixed_noise": fixed_noise,
        "img_progress": img_progress,
    }
    if gamma_optim:
        ckpt["gamma_optim"] = gamma_optim.state_dict()
    if g_losses_perceptual:
        ckpt["g_losses_perceptual"] = g_losses_perceptual
    if g_losses_GAN:
        ckpt["g_losses_GAN"] = g_losses_GAN
    if channel_number:
        ckpt["channel_number"] = channel_number
    if generator_lr_scheduler:
        ckpt["generator_lr_scheduler"] = generator_lr_scheduler.state_dict()
    if discriminator_lr_scheduler:
        ckpt["discriminator_lr_scheduler"] = discriminator_lr_scheduler.state_dict()
    if gamma_lr_scheduler:
        ckpt["gamma_lr_scheduler"] = gamma_lr_scheduler.state_dict()
    torch.save(ckpt, path)
    print("Saved checkpoint file at {}".format(path))


def load_checkpoint(
        path,
        generator,
        discriminator=None,
        generator_optim=None,
        discriminator_optim=None,
        gamma_optim=None,
        generator_lr_scheduler=None,
        discriminator_lr_scheduler=None,
        gamma_lr_scheduler=None,
):
    if not os.path.isfile(path):
        raise Exception(f"No such file: {path}")
    print(f"Loading checkpoint from {path}")

    ckpt = torch.load(path)
    generator.load_state_dict(ckpt["generator"])
    if discriminator:
        discriminator.load_state_dict(ckpt["discriminator"])
    if generator_optim:
        generator_optim.load_state_dict(ckpt["generator_optim"])
    if discriminator_optim:
        discriminator_optim.load_state_dict(ckpt["discriminator_optim"])
    if gamma_optim:
        gamma_optim.load_state_dict(ckpt["gamma_optim"])
    if generator_lr_scheduler:
        generator_lr_scheduler.load_state_dict(ckpt["generator_lr_scheduler"])
    if discriminator_lr_scheduler:
        discriminator_lr_scheduler.load_state_dict(ckpt["discriminator_lr_scheduler"])
    if gamma_lr_scheduler:
        gamma_lr_scheduler.load_state_dict(ckpt["gamma_lr_scheduler"])
    return ckpt


def find_largest_divisor(num_channels, max_groups):
    for n in range(min(num_channels, max_groups), 0, -1):
        if num_channels % n == 0:
            return n
    return 1  # fallback of 1 when no divisor is found


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def soft_threshold(w, th):
    """
    Soft-sign function. If the absolute value of w is below th, then zero channel (channel pruning).
    :param w: weight.
    :param th: threshold value.
    :return: weighted threshold value.
    """
    with torch.no_grad():
        temp = torch.abs(w) - th
        return torch.sign(w) * nn.functional.relu(temp)


def compute_perceptual_loss(vgg1, vgg2, beta=1e5, layer="relu1_2"):
    """
    Compute perceptual loss between two vgg features.
    :param vgg1: first vgg feature.
    :param vgg2: second vgg feature.
    :param beta: loss style coefficient.
    :param layer:
    :return: perceptual loss, the loss in content, and the loss in style
    """

    loss_content = 0
    if layer == "relu1_2":
        loss_content = L1Loss()(vgg1.relu1_2, vgg2.relu1_2)
    elif layer == "relu2_2":
        loss_content = L1Loss()(vgg1.relu2_2, vgg2.relu2_2)
    elif layer == "relu3_3":
        loss_content = L1Loss()(vgg1.relu3_3, vgg2.relu3_3)
    loss_style = 0
    for _, (vf_g, vf_c) in enumerate(zip(vgg1, vgg2)):
        gm_g, gm_c = gram_matrix(vf_g), gram_matrix(vf_c)
        loss_style += nn.functional.mse_loss(gm_g, gm_c)
    loss_perceptual = loss_content + beta * loss_style
    return loss_perceptual, loss_content, loss_style


def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


count_ops = 0
num_ids = 0


def num_ops_hook(self, _input, _output):
    global count_ops, num_ids
    count_ops += (self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * _output.size(2)
                  * _output.size(3) / self.groups)
    num_ids += 1


def measure_model(model: nn.Module, input_shape: tuple):
    global count_ops, num_ids
    count_ops, num_ids = 0, 0  # Reset counts

    hooks = []
    for module in model.named_modules():
        if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.ConvTranspose2d):
            hooks.append(module[1].register_forward_hook(num_ops_hook))

    _ = model(torch.randn(input_shape))  # Forward pass

    # Remove hooks after measurement to prevent memory leak
    for hook in hooks:
        hook.remove()

    return count_ops
