import argparse
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import wandb


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # Project Info
    entity: str = "tonykorea99-dankook-university"
    project: str = "VAE_MNIST_PRAC"
    run_name: str = ""
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 0
    val_ratio: float = 0.1

    # Model
    latent_dim: int = 16

    # Training
    epochs: int = 200
    lr: float = 1e-3
    beta: float = 1.0
    loss: str = "mse"  # 'mse' or 'bce'

    # Logging / Eval
    log_every: int = 50
    eval_every: int = 50
    val_eval_batches: int = 0  # 0 means full validation

    # Early Stopping / Overfit Protection
    patience: int = 3
    min_delta: float = 1e-4
    train_min_improve: float = 0.0

    # Image Logging
    num_sample_images: int = 64
    num_recon_images: int = 32
    image_every_epochs: int = 1


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 2. Model: ConvVAE
# -----------------------------------------------------------------------------
class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # Encoder: (1, 28, 28) -> (64, 7, 7)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (64, 7, 7)
            nn.ReLU(True),
        )
        self.enc_feat_dim = 64 * 7 * 7
        
        # Latent Space
        self.fc_mu = nn.Linear(self.enc_feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_feat_dim, latent_dim)

        # Decoder: Latent -> (1, 28, 28)
        self.fc_dec = nn.Linear(latent_dim, self.enc_feat_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> (32, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> (1, 28, 28)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(h.size(0), 64, 7, 7)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, 
             loss_type: str, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes VAE Loss (Recon + Beta * KL)"""
    if loss_type == "mse":
        recon = F.mse_loss(x_hat, x, reduction="sum")
    elif loss_type == "bce":
        recon = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total = recon + beta * kl
    return total, recon, kl


# -----------------------------------------------------------------------------
# 3. Helpers: Data & Early Stopping
# -----------------------------------------------------------------------------
def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=transform)

    val_size = int(len(dataset) * cfg.val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    use_pin = cfg.device.startswith("cuda")
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=cfg.num_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                            num_workers=cfg.num_workers, pin_memory=use_pin)
    
    return train_loader, val_loader


class EarlyStopper:
    """Handles logic for stopping training if validation metric stops improving."""
    def __init__(self, patience: int, min_delta: float, train_min_improve: float):
        self.patience = patience
        self.min_delta = min_delta
        self.train_min_improve = train_min_improve
        
        self.best_val_loss = float('inf')
        self.no_improve_count = 0
        self.prev_train_loss = None
        self.stop_reason = ""
        self.best_state = {}

    def check(self, val_loss: float, train_loss: float, step_info: Dict[str, Any]) -> bool:
        """
        Returns True if training should stop.
        Updates internal state (best loss, counters).
        """
        # 1. Validation Improved
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.best_state = step_info
            self.no_improve_count = 0
            
            wandb.log({
                "best/val_loss": val_loss,
                "best/epoch": step_info["epoch"],
                "best/global_step": step_info["global_step"]
            }, step=step_info["global_step"])
            return False

        # 2. Validation NOT Improved
        # Check if training is still strictly improving (Generalization Gap check)
        train_is_improving = False
        if self.prev_train_loss is not None:
            train_is_improving = train_loss < (self.prev_train_loss - self.train_min_improve)
        
        self.prev_train_loss = train_loss

        # Overfitting signal: Val bad AND Train good
        if train_is_improving:
            self.no_improve_count += 1
        else:
            # If train is also stuck, we don't necessarily count it as "overfitting" 
            # (could be underfitting or plateau), so we reset or hold. 
            # Following original logic: reset if train not improving
            self.no_improve_count = 0 

        print(f"  [EarlyStop] No improve: {self.no_improve_count}/{self.patience} | Val: {val_loss:.4f} (Best: {self.best_val_loss:.4f})")

        if self.no_improve_count >= self.patience:
            self.stop_reason = f"Overfitting: Val not improved for {self.patience} checks while Train improved."
            return True
            
        return False


# -----------------------------------------------------------------------------
# 4. Trainer Class
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: Config, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 train_loader: DataLoader, val_loader: DataLoader):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.early_stopper = EarlyStopper(cfg.patience, cfg.min_delta, cfg.train_min_improve)
        self.global_step = 0
        
        # Fixed input for consistent image logging
        self.fixed_x, _ = next(iter(val_loader))
        self.fixed_x = self.fixed_x.to(cfg.device)

    def train(self):
        print(f"[Info] Starting training on {self.cfg.device}...")
        
        for epoch in range(self.cfg.epochs):
            print(f"\n[Epoch {epoch+1}/{self.cfg.epochs}] START")
            stop_signal = self._run_epoch(epoch)
            
            # Log images periodically or at the end
            if ((epoch + 1) % self.cfg.image_every_epochs == 0) or stop_signal:
                self._log_images(step=self.global_step)
            
            if stop_signal:
                print(f"\n[STOP] {self.early_stopper.stop_reason}")
                self._log_stop_info()
                break

        self._save_checkpoint()
        self._print_summary()

    def _run_epoch(self, epoch: int) -> bool:
        self.model.train()
        win_loss_sum = 0.0
        win_count = 0

        for i, (x, _) in enumerate(self.train_loader):
            self.global_step += 1
            x = x.to(self.cfg.device)
            bsz = x.size(0)

            # Forward & Backward
            x_hat, mu, logvar = self.model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, self.cfg.loss, self.cfg.beta)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # Tracking window for early stopping
            win_loss_sum += loss.item()
            win_count += bsz

            # Logging
            if (i + 1) % self.cfg.log_every == 0:
                self._log_metrics(loss, recon, kl, epoch, i)

            # Evaluation & Early Stopping Check
            if (i + 1) % self.cfg.eval_every == 0:
                train_loss_avg = win_loss_sum / win_count
                val_loss, val_recon, val_kl = self._validate()
                
                # Log Validation
                wandb.log({
                    "val/loss": val_loss, "val/recon": val_recon, "val/kl": val_kl,
                    "epoch": epoch
                }, step=self.global_step)

                # Check Early Stopping
                step_info = {"epoch": epoch, "step": i + 1, "global_step": self.global_step}
                should_stop = self.early_stopper.check(val_loss, train_loss_avg, step_info)
                
                if should_stop:
                    return True
                
                # Reset window
                win_loss_sum = 0.0
                win_count = 0
                self.model.train() # Switch back to train mode

        return False

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        sum_total, sum_recon, sum_kl = 0.0, 0.0, 0.0
        seen = 0
        
        for i, (x, _) in enumerate(self.val_loader):
            if self.cfg.val_eval_batches > 0 and i >= self.cfg.val_eval_batches:
                break
            
            x = x.to(self.cfg.device)
            x_hat, mu, logvar = self.model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, self.cfg.loss, self.cfg.beta)

            sum_total += loss.item()
            sum_recon += recon.item()
            sum_kl += kl.item()
            seen += x.size(0)

        return sum_total / seen, sum_recon / seen, sum_kl / seen

    def _log_metrics(self, loss, recon, kl, epoch, step_in_epoch):
        bsz = self.cfg.batch_size # Approximate
        wandb.log({
            "train/loss": loss.item() / bsz,
            "train/recon": recon.item() / bsz,
            "train/kl": kl.item() / bsz,
            "train/lr": self.optimizer.param_groups[0]["lr"],
            "epoch": epoch
        }, step=self.global_step)
        print(f"  [Ep {epoch+1} - {step_in_epoch+1}] Train Loss: {loss.item()/bsz:.4f}")

    @torch.no_grad()
    def _log_images(self, step: int):
        self.model.eval()
        
        # 1. Reconstruction
        n = min(self.cfg.num_recon_images, self.fixed_x.size(0))
        x_in = self.fixed_x[:n]
        x_out, _, _ = self.model(x_in)
        
        if self.cfg.loss == "bce":
            x_out = torch.sigmoid(x_out)
        
        pairs = torch.cat([x_in.cpu(), x_out.cpu().clamp(0, 1)], dim=0)
        grid_recon = make_grid(pairs, nrow=n, padding=2)

        # 2. Sampling
        z = torch.randn(self.cfg.num_sample_images, self.cfg.latent_dim, device=self.cfg.device)
        samples = self.model.decode(z)
        if self.cfg.loss == "bce":
            samples = torch.sigmoid(samples)
        
        grid_sample = make_grid(samples.cpu().clamp(0, 1), nrow=int(self.cfg.num_sample_images**0.5), padding=2)

        wandb.log({
            "images/recon": wandb.Image(grid_recon, caption="Original vs Recon"),
            "images/sample": wandb.Image(grid_sample, caption="Random Generation")
        }, step=step)

    def _log_stop_info(self):
        wandb.log({
            "final/stop_reason": self.early_stopper.stop_reason,
            "final/stop_global_step": self.global_step
        }, step=self.global_step)

    def _save_checkpoint(self):
        Path("checkpoints").mkdir(exist_ok=True)
        filename = f"checkpoints/convvae_{self.cfg.loss}_lat{self.cfg.latent_dim}.pt"
        
        torch.save({
            "model_state": self.model.state_dict(),
            "config": asdict(self.cfg),
            "best_val_loss": self.early_stopper.best_val_loss
        }, filename)
        wandb.save(filename)
        print(f"\n[Info] Checkpoint saved to {filename}")

    def _print_summary(self):
        print("\n" + "="*40)
        print(" TRAINING SUMMARY")
        print("="*40)
        if self.early_stopper.best_state:
            b = self.early_stopper.best_state
            print(f"Best Model @ Epoch {b['epoch']+1}, Step {b['step']}")
            print(f"Best Val Loss: {self.early_stopper.best_val_loss:.5f}")
        else:
            print("No validation improvement recorded.")
        print(f"Stop Reason: {self.early_stopper.stop_reason or 'Max Epochs Reached'}")
        print("="*40 + "\n")


# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------
def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    # Define only necessary override arguments here
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--loss", type=str, choices=["mse", "bce"], default="mse")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Update Config with args
    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            if isinstance(v, str) and v == "": continue # skip empty strings if default is set
            setattr(cfg, k, v)
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    
    # Init WandB
    wandb.init(entity=cfg.entity, project=cfg.project, name=cfg.run_name or None, config=asdict(cfg))

    # Data & Model
    train_loader, val_loader = build_dataloaders(cfg)
    model = ConvVAE(latent_dim=cfg.latent_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Train
    trainer = Trainer(cfg, model, optimizer, train_loader, val_loader)
    trainer.train()
    
    wandb.finish()


if __name__ == "__main__":
    main()
