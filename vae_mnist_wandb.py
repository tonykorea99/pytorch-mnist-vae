
import argparse
import os
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import wandb



@dataclass
class Config:

    entity: str = "tonykorea99-dankook-university"
    project: str = "VAE_MNIST_PRAC"

 
    run_name: str = ""

   
    seed: int = 42

    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    val_ratio: float = 0.1

 
    latent_dim: int = 16
    hidden_dim: int = 512


    epochs: int = 10
    lr: float = 1e-3
    beta: float = 1.0
    loss: str = "mse"  # default: MSE
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

  
    log_every: int = 50
    num_sample_images: int = 64
    num_recon_images: int = 32


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_mnist(x: torch.Tensor) -> torch.Tensor:
    """(B,1,28,28) -> (B,784)"""
    return x.view(x.size(0), -1)



class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        # Encoder: x -> h
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )

        # Encoder heads: μ, log(σ^2)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> x_hat (linear output for MSE)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    loss_type: str,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reconstruction loss
    if loss_type == "mse":
        recon = F.mse_loss(x_hat, x, reduction="sum")
    elif loss_type == "bce":
        # 참고용: decoder가 선형 출력일 때 logits 기반 BCE
        recon = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
    else:
        raise ValueError("loss_type은 'mse' 또는 'bce'만 가능합니다.")

    # KL(q(z|x) || p(z)) with p(z)=N(0,I)
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)

    total = recon + beta * kl
    return total, recon, kl


def forward_and_loss(
    model: VAE,
    cfg: Config,
    x_img: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    공통 계산:
    - 입력 MNIST 이미지를 flatten
    - forward
    - ELBO(loss, recon, kl) 계산
    """
    x = flatten_mnist(x_img).to(cfg.device)
    x_hat, mu, logvar = model(x)
    return vae_loss(x, x_hat, mu, logvar, cfg.loss, cfg.beta)


@torch.no_grad()
def log_images(model: VAE, cfg: Config, x_batch: torch.Tensor, step: int) -> None:
    model.eval()

    # (A) Recon pairs
    n = min(cfg.num_recon_images, x_batch.size(0))
    x = x_batch[:n].to(cfg.device)
    x_flat = flatten_mnist(x)

    x_hat, _, _ = model(x_flat)
    x_hat_img = x_hat.view(-1, 1, 28, 28).clamp(0, 1).cpu()  # 시각화용 clamp
    orig = x.cpu()

    pairs = torch.cat([orig, x_hat_img], dim=0)
    grid_recon = make_grid(pairs, nrow=n, padding=2)
    wandb.log(
        {"images/recon_pairs": wandb.Image(grid_recon, caption="Top: original / Bottom: reconstruction (clamped)")},
        step=step,
    )

    # (B) Random samples from prior
    z = torch.randn(cfg.num_sample_images, cfg.latent_dim, device=cfg.device)
    samples = model.decode(z).view(-1, 1, 28, 28).clamp(0, 1).cpu()

    side = int(cfg.num_sample_images ** 0.5)
    grid_samples = make_grid(samples, nrow=side, padding=2)
    wandb.log(
        {"images/samples": wandb.Image(grid_samples, caption="Samples from z~N(0,I) (clamped)")},
        step=step,
    )


def train_one_epoch(model: VAE, opt: torch.optim.Optimizer, loader: DataLoader, cfg: Config, epoch: int):
    model.train()

    sum_total = 0.0
    sum_recon = 0.0
    sum_kl = 0.0

    for i, (x_img, _) in enumerate(loader):
        # forward + loss
        loss, recon, kl = forward_and_loss(model, cfg, x_img)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        sum_total += loss.item()
        sum_recon += recon.item()
        sum_kl += kl.item()

        # batch log (기존과 동일한 step 정의)
        global_step = epoch * len(loader) + i
        if (i + 1) % cfg.log_every == 0:
            bsz = x_img.size(0)
            wandb.log(
                {
                    "train/loss": loss.item() / bsz,
                    "train/recon": recon.item() / bsz,
                    "train/kl": kl.item() / bsz,
                    "train/beta": cfg.beta,
                    "train/lr": opt.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

    n_data = len(loader.dataset)
    return sum_total / n_data, sum_recon / n_data, sum_kl / n_data


@torch.no_grad()
def eval_one_epoch(model: VAE, loader: DataLoader, cfg: Config):
    model.eval()

    sum_total = 0.0
    sum_recon = 0.0
    sum_kl = 0.0

    for x_img, _ in loader:
        loss, recon, kl = forward_and_loss(model, cfg, x_img)
        sum_total += loss.item()
        sum_recon += recon.item()
        sum_kl += kl.item()

    n_data = len(loader.dataset)
    return sum_total / n_data, sum_recon / n_data, sum_kl / n_data


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=tfm)

    val_len = int(len(ds) * cfg.val_ratio)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_model_and_optimizer(cfg: Config) -> Tuple[VAE, torch.optim.Optimizer]:
    model = VAE(input_dim=28 * 28, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    return model, opt


def save_checkpoint(model: VAE, cfg: Config) -> str:
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/vae_mnist_mse_lat{cfg.latent_dim}_beta{cfg.beta}.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
    wandb.save(ckpt_path)
    return ckpt_path


def init_wandb(cfg: Config) -> None:
    wandb.init(
        entity=cfg.entity,
        project=cfg.project,
        name=cfg.run_name if cfg.run_name else None,
        config=asdict(cfg),
    )


# ----------------------------
# 7) main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--loss", type=str, choices=["mse", "bce"], default="mse")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--run_name", type=str, default="")

    args = parser.parse_args()

    cfg = Config(
        run_name=args.run_name,
        seed=args.seed,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        loss=args.loss,
        log_every=args.log_every,
    )

    set_seed(cfg.seed)
    init_wandb(cfg)

    train_loader, val_loader = build_dataloaders(cfg)
    model, opt = build_model_and_optimizer(cfg)

    
    fixed_x, _ = next(iter(val_loader))

   
    for epoch in range(cfg.epochs):
        tr_loss, tr_recon, tr_kl = train_one_epoch(model, opt, train_loader, cfg, epoch)
        va_loss, va_recon, va_kl = eval_one_epoch(model, val_loader, cfg)

    
        step = (epoch + 1) * len(train_loader)

        wandb.log(
            {
                "epoch": epoch,
                "epoch/train_loss": tr_loss,
                "epoch/train_recon": tr_recon,
                "epoch/train_kl": tr_kl,
                "epoch/val_loss": va_loss,
                "epoch/val_recon": va_recon,
                "epoch/val_kl": va_kl,
            },
            step=step,
        )

        log_images(model, cfg, fixed_x, step=step)

    save_checkpoint(model, cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
