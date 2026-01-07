# -*- coding: utf-8 -*-
"""
CNN(Conv-VAE) only + W&B logging + Overfitting stop + Console progress

핵심 기능:
1) MLP 제거 -> ConvVAE만 사용
2) 학습 중 주기적으로 validation을 돌려서
   - best val(=generalization 최고 지점) 기록
   - "train은 계속 좋아지는데(val best 갱신 못하고 악화)"가 연속으로 나타나면 오버피팅으로 판단 -> 즉시 정지
   - 정지 시 "현재 epoch(블락)에서 step 몇 개까지 돌았는지" 출력
   - best val이 발생한 epoch/step도 출력 (네가 말한 맥시마이즈 지점)
3) 콘솔에 epoch/step 진행상황 계속 출력

실행 예:
  python vae_mnist_wandb_cnn_overfitstop.py --epochs 500 --eval_every 50 --patience 3 --num_workers 0
"""

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


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    entity: str = "tonykorea99-dankook-university"
    project: str = "VAE_MNIST_PRAC"
    run_name: str = ""

    seed: int = 42

    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 0          # Windows 안정성: 기본 0 추천
    val_ratio: float = 0.1

    latent_dim: int = 16

    # --- Train ---
    epochs: int = 200
    lr: float = 1e-3
    beta: float = 1.0
    loss: str = "mse"  # "mse" or "bce"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Logging / Eval ---
    log_every: int = 50          # train 배치 로그 주기
    eval_every: int = 50         # "오버피팅 감지"를 위한 val 평가 주기(steps)
    val_eval_batches: int = 0    # 0이면 val 전체로 평가(정확). >0이면 그 배치 수만 평가(빠름)

    # --- Overfit Stop ---
    patience: int = 3            # 연속으로 val best 갱신 실패(+train 개선) 몇 번이면 stop
    min_delta: float = 1e-4      # val 개선으로 인정할 최소 폭
    train_min_improve: float = 0.0  # train이 좋아졌다고 보는 최소 폭(0이면 '조금이라도' 감소)

    # --- Image logging ---
    num_sample_images: int = 64
    num_recon_images: int = 32
    image_every_epochs: int = 1  # 몇 epoch마다 이미지 로그


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# CNN VAE (Conv-VAE only)
# ----------------------------
class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # Encoder: (B,1,28,28) -> (B,64,7,7)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(True),
        )
        enc_feat = 64 * 7 * 7
        self.fc_mu = nn.Linear(enc_feat, latent_dim)
        self.fc_logvar = nn.Linear(enc_feat, latent_dim)

        # Decoder: z -> (B,1,28,28)
        self.fc_dec = nn.Linear(latent_dim, enc_feat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 14 -> 28
            # sigmoid는 넣지 않음:
            # - MSE: raw OK
            # - BCE: logits로 보고 BCEWithLogits 사용
        )

    def encode(self, x_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x_img)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), 64, 7, 7)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x_img)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# ----------------------------
# ELBO loss = recon + beta*KL
# ----------------------------
def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    loss_type: str,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss_type == "mse":
        recon = F.mse_loss(x_hat, x, reduction="sum")
    elif loss_type == "bce":
        recon = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
    else:
        raise ValueError("loss는 'mse' 또는 'bce'만 가능합니다.")

    # KL(q(z|x)||p(z)), p(z)=N(0,I)
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)
    total = recon + beta * kl
    return total, recon, kl


@torch.no_grad()
def _postprocess_for_viz(t: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "bce":
        t = torch.sigmoid(t)
    return t.clamp(0, 1)


@torch.no_grad()
def log_images(model: nn.Module, cfg: Config, x_batch: torch.Tensor, step: int) -> None:
    model.eval()

    # Recon pairs
    n = min(cfg.num_recon_images, x_batch.size(0))
    x = x_batch[:n].to(cfg.device)
    x_hat, _, _ = model(x)
    x_hat = _postprocess_for_viz(x_hat, cfg.loss).cpu()
    orig = x.cpu()

    pairs = torch.cat([orig, x_hat], dim=0)
    grid_recon = make_grid(pairs, nrow=n, padding=2)
    wandb.log(
        {"images/recon_pairs": wandb.Image(grid_recon, caption="Top: original / Bottom: reconstruction")},
        step=step,
    )

    # Samples from prior
    z = torch.randn(cfg.num_sample_images, cfg.latent_dim, device=cfg.device)
    samples = model.decode(z)
    samples = _postprocess_for_viz(samples, cfg.loss).cpu()
    side = int(cfg.num_sample_images ** 0.5)
    grid_samples = make_grid(samples, nrow=side, padding=2)
    wandb.log(
        {"images/samples": wandb.Image(grid_samples, caption="Samples from z~N(0,I)")},
        step=step,
    )


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.ToTensor()
    ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=tfm)

    val_len = int(len(ds) * cfg.val_ratio)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    pin = True if cfg.device.startswith("cuda") else False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


@torch.no_grad()
def eval_val_loss(model: nn.Module, val_loader: DataLoader, cfg: Config) -> Tuple[float, float, float]:
    """val 전체(or 일부 배치)로 평균 loss(recon,kl) 계산"""
    model.eval()

    sum_total = 0.0
    sum_recon = 0.0
    sum_kl = 0.0
    seen = 0

    for bi, (x, _) in enumerate(val_loader):
        x = x.to(cfg.device)
        x_hat, mu, logvar = model(x)
        total, recon, kl = vae_loss(x, x_hat, mu, logvar, cfg.loss, cfg.beta)

        sum_total += total.item()
        sum_recon += recon.item()
        sum_kl += kl.item()
        seen += x.size(0)

        if cfg.val_eval_batches > 0 and (bi + 1) >= cfg.val_eval_batches:
            break

    # per-sample average
    return sum_total / seen, sum_recon / seen, sum_kl / seen


def init_wandb(cfg: Config) -> None:
    wandb.init(
        entity=cfg.entity,
        project=cfg.project,
        name=cfg.run_name if cfg.run_name else None,
        config=asdict(cfg),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--loss", type=str, choices=["mse", "bce"], default="mse")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--val_eval_batches", type=int, default=0)

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--train_min_improve", type=float, default=0.0)

    parser.add_argument("--run_name", type=str, default="")

    args = parser.parse_args()

    cfg = Config(
        run_name=args.run_name,
        seed=args.seed,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,

        latent_dim=args.latent_dim,

        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        loss=args.loss,

        log_every=args.log_every,
        eval_every=args.eval_every,
        val_eval_batches=args.val_eval_batches,

        patience=args.patience,
        min_delta=args.min_delta,
        train_min_improve=args.train_min_improve,
    )

    set_seed(cfg.seed)
    init_wandb(cfg)

    train_loader, val_loader = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)

    print(f"[Info] device={cfg.device}")
    print(f"[Info] steps_per_epoch(one block) = {steps_per_epoch} (max i = {steps_per_epoch-1})")
    print(f"[Info] train_size={len(train_loader.dataset)}, val_size={len(val_loader.dataset)}, batch_size={cfg.batch_size}")
    wandb.log({"meta/steps_per_epoch": steps_per_epoch}, step=0)

    model = ConvVAE(latent_dim=cfg.latent_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 고정 배치(이미지 로깅용)
    fixed_x, _ = next(iter(val_loader))

    # ---- Overfit tracking ----
    best_val = float("inf")
    best_info = {"epoch": None, "step_in_epoch": None, "global_step": None, "val_loss": None}

    no_improve = 0
    prev_train_window: Optional[float] = None

    stop = False
    stop_info = {"epoch": None, "step_in_epoch": None, "global_step": None, "reason": ""}

    global_step = 0

    for epoch in range(cfg.epochs):
        print(f"\n[Epoch {epoch+1}/{cfg.epochs}] START")
        model.train()

        # window sums (eval_every마다 train 평균을 계산해서 'train은 좋아지는 중인지' 판단)
        win_sum_total = 0.0
        win_seen = 0

        for i, (x, _) in enumerate(train_loader):
            x = x.to(cfg.device)
            x_hat, mu, logvar = model(x)
            total, recon, kl = vae_loss(x, x_hat, mu, logvar, cfg.loss, cfg.beta)

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

            # per-sample for window tracking
            bsz = x.size(0)
            win_sum_total += total.item()
            win_seen += bsz

            # train batch logging
            if (i + 1) % cfg.log_every == 0:
                train_loss_ps = total.item() / bsz
                wandb.log(
                    {
                        "train/loss": train_loss_ps,
                        "train/recon": recon.item() / bsz,
                        "train/kl": kl.item() / bsz,
                        "train/beta": cfg.beta,
                        "train/lr": opt.param_groups[0]["lr"],
                        "epoch": epoch,
                        "meta/step_in_epoch": i + 1,
                    },
                    step=global_step,
                )
                print(f"[Epoch {epoch+1}] step {i+1}/{steps_per_epoch} | train_loss(ps)={train_loss_ps:.4f}")

            # ----- periodic validation for overfit detection -----
            if (i + 1) % cfg.eval_every == 0:
                # train window avg (per-sample)
                train_window = (win_sum_total / win_seen) if win_seen > 0 else None

                val_total, val_recon, val_kl = eval_val_loss(model, val_loader, cfg)

                wandb.log(
                    {
                        "val/step_loss": val_total,
                        "val/step_recon": val_recon,
                        "val/step_kl": val_kl,
                        "meta/step_in_epoch": i + 1,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

                print(
                    f"  [VAL @ epoch {epoch+1}, step {i+1}] "
                    f"train_window={train_window:.4f} | val_loss={val_total:.4f} | best_val={best_val:.4f} | no_improve={no_improve}"
                )

                # best val 갱신?
                if val_total < (best_val - cfg.min_delta):
                    best_val = val_total
                    best_info = {
                        "epoch": epoch,
                        "step_in_epoch": i + 1,
                        "global_step": global_step,
                        "val_loss": val_total,
                    }
                    no_improve = 0
                    wandb.log(
                        {
                            "best/val_loss": best_val,
                            "best/epoch": epoch,
                            "best/step_in_epoch": i + 1,
                            "best/global_step": global_step,
                        },
                        step=global_step,
                    )
                else:
                    # train이 개선 중인지 확인(=오버피팅 조건에 필요)
                    train_improving = False
                    if prev_train_window is not None and train_window is not None:
                        train_improving = (train_window < (prev_train_window - cfg.train_min_improve))

                    # "val은 개선 안 됨 + train은 개선 중"이면 오버피팅 신호로 카운트
                    if train_improving:
                        no_improve += 1
                    else:
                        # train도 안 좋아지면(plateau 등) 오버피팅이라 보기 애매해서 카운트 리셋
                        no_improve = 0

                prev_train_window = train_window

                # window reset
                win_sum_total = 0.0
                win_seen = 0

                # stop 조건
                if no_improve >= cfg.patience and best_info["epoch"] is not None:
                    stop = True
                    stop_info = {
                        "epoch": epoch,
                        "step_in_epoch": i + 1,
                        "global_step": global_step,
                        "reason": f"Overfitting detected: val not improving for {cfg.patience} evals while train improves",
                    }
                    wandb.log(
                        {
                            "stop/epoch": epoch,
                            "stop/step_in_epoch": i + 1,
                            "stop/global_step": global_step,
                            "stop/reason": stop_info["reason"],
                        },
                        step=global_step,
                    )
                    print("\n[STOP] Overfitting detected -> training halted.")
                    break

            global_step += 1

        # epoch end logging (조용했던 문제 해결: epoch 단위 출력)
        # (주의: 중간 stop이면 epoch 완주 안 했을 수 있음)
        if best_info["epoch"] is not None:
            print(
                f"[Epoch {epoch+1}] END | current_best_val={best_val:.4f} "
                f"(best at epoch {best_info['epoch']+1}, step {best_info['step_in_epoch']}, global {best_info['global_step']})"
            )

        # 이미지 로깅 (epoch 단위)
        if ((epoch + 1) % cfg.image_every_epochs == 0) and (not stop):
            log_images(model, cfg, fixed_x, step=global_step)

        if stop:
            # stop 시점에도 이미지 1번 찍어줌(원하면 제거 가능)
            log_images(model, cfg, fixed_x, step=global_step)
            break

    # ----------------------------
    # Final summary (콘솔 + W&B)
    # ----------------------------
    print("\n==================== SUMMARY ====================")
    print(f"steps_per_epoch(one block) = {steps_per_epoch}")
    if best_info["epoch"] is not None:
        print(
            f"[BEST (maximize point)] epoch={best_info['epoch']+1} | step_in_epoch={best_info['step_in_epoch']} "
            f"| global_step={best_info['global_step']} | best_val_loss={best_info['val_loss']:.6f}"
        )
    if stop_info["epoch"] is not None:
        print(
            f"[STOP (overfit)] epoch={stop_info['epoch']+1} | steps_done_in_that_block={stop_info['step_in_epoch']} "
            f"| global_step={stop_info['global_step']} | reason={stop_info['reason']}"
        )
    else:
        print("[STOP] not triggered (training reached max epochs)")
    print("=================================================\n")

    wandb.log(
        {
            "final/steps_per_epoch": steps_per_epoch,
            "final/best_val_loss": best_info["val_loss"] if best_info["val_loss"] is not None else None,
            "final/best_epoch": (best_info["epoch"] if best_info["epoch"] is not None else None),
            "final/best_step_in_epoch": best_info["step_in_epoch"],
            "final/stop_epoch": (stop_info["epoch"] if stop_info["epoch"] is not None else None),
            "final/stop_step_in_epoch": stop_info["step_in_epoch"],
            "final/stop_global_step": stop_info["global_step"],
            "final/stop_reason": stop_info["reason"],
        },
        step=global_step,
    )

    # checkpoint 저장(선택)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/convvae_mnist_{cfg.loss}_lat{cfg.latent_dim}_beta{cfg.beta}.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
    wandb.save(ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()
