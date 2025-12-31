# -*- coding: utf-8 -*-
"""
Diffusion.ipynb의 사본 (Windows + CUDA용 .py 변환)
- 원문 학습/생성 흐름은 유지
- 노트북 전용(!, apt, megadl)만 윈도우에서 실행 가능한 형태로 치환
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def sh(cmd: str):
    """Notebook의 ! 커맨드를 파이썬에서 실행하기 위한 헬퍼."""
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def exists_cmd(name: str) -> bool:
    return shutil.which(name) is not None

# ---------------------------
# 0) (선택) GPU/드라이버 체크
# ---------------------------
if exists_cmd("nvidia-smi"):
    sh("nvidia-smi")
else:
    print("[INFO] nvidia-smi not found. (드라이버/경로 확인 필요할 수 있음)")

# ---------------------------
# 1) PyTorch 환경 체크 (강제 다운그레이드 X)
# ---------------------------
import torch
print("[INFO] torch =", torch.__version__)
print("[INFO] torch.cuda.is_available() =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] GPU:", torch.cuda.get_device_name(0))

# ---------------------------
# 2) 레포 클론 + import path 설정
# ---------------------------
REPO_DIR = Path("./mnist_generative_practice")

if not REPO_DIR.exists():
    if not exists_cmd("git"):
        raise RuntimeError("git이 필요합니다. Git 설치 후 다시 실행하세요.")
    sh(f'git clone https://github.com/shhommychon/mnist_generative_practice.git "{REPO_DIR}"')

sys.path = [str(REPO_DIR)] + sys.path

# ---------------------------
# 3) MEGA 가중치 파일 준비 (윈도우)
#    - 옵션 A: MegaCMD 설치 후 mega-get 사용(자동)
#    - 옵션 B: 수동 다운로드 후 같은 폴더에 두기
# ---------------------------
def ensure_mega_file(url: str, filename: str):
    file_path = Path(filename)
    if file_path.exists():
        print(f"[INFO] Found: {filename}")
        return

    # MegaCMD가 설치되어 있으면 public link로 다운로드 가능
    # (명령어 이름은 보통 mega-get)
    if exists_cmd("mega-get"):
        print(f"[INFO] Downloading via mega-get: {filename}")
        sh(f'mega-get "{url}"')
    else:
        # 수동 다운로드 안내 (코드 실행은 중단)
        raise FileNotFoundError(
            f"\n[ERROR] {filename} 파일이 없습니다.\n"
            f"아래 MEGA 링크에서 파일을 수동으로 다운로드해서 현재 폴더에 넣어주세요:\n"
            f"{url}\n\n"
            f"또는 MegaCMD를 설치해 mega-get이 PATH에 잡히게 한 뒤 다시 실행하세요."
        )

# mnist_feature_extractor.dth
ensure_mega_file(
    "https://mega.nz/file/DgUQyDyB#7Gyq_9kzCz8FcGZV659VD1Cq1_36wimGVOG2Eram3P8",
    "mnist_feature_extractor.dth",
)

# mnist_classifier.dth
ensure_mega_file(
    "https://mega.nz/file/H49S3bTI#qsonzlkV3JMniTbyzV77BB9VLhwmh1OJLTgxuD4PEMM",
    "mnist_classifier.dth",
)

# ---------------------------
# 4) (원문) 실습 코드
# ---------------------------
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터셋 다운로드 및 불러오기
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset = datasets.MNIST(root='./data', train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

mnist_valset, mnist_testset = torch.utils.data.random_split(
                                mnist_testset,
                                [
                                    int(0.9 * len(mnist_testset)),
                                    int(0.1 * len(mnist_testset))
                                ]
                            )

# Windows는 멀티프로세스 DataLoader가 가끔 귀찮게 굴 수 있어서 num_workers=0 추천
train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True, num_workers=0)
val_dataloader   = torch.utils.data.DataLoader(mnist_valset,   batch_size=32, shuffle=False, num_workers=0)
test_dataloader  = torch.utils.data.DataLoader(mnist_testset,  batch_size=32, shuffle=False, num_workers=0)

print("Training dataset size: ", len(mnist_trainset))
print("Validation dataset size: ", len(mnist_valset))
print("Testing dataset size: ", len(mnist_testset))

# Diffusion model 학습 보조용 사전학습 MNIST 분류기
from models.classifier import MNISTClassifier

c_model = MNISTClassifier()
c_model.feature_extractor.load_state_dict(torch.load("mnist_feature_extractor.dth", map_location="cpu"))
c_model.classifier.load_state_dict(torch.load("mnist_classifier.dth", map_location="cpu"))

if torch.cuda.is_available():
    c_model.cuda()

# freeze
for param in c_model.parameters():
    param.requires_grad = False

from models.diffusion import MNISTDiffusion

model = MNISTDiffusion()
if torch.cuda.is_available():
    model.cuda()

pixelwise_loss = nn.L1Loss()
classification_loss = nn.CrossEntropyLoss()

lr = 0.0002
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

max_epochs = 100
best_val_loss = 100
max_patience = 5

# 이미지 생성 코드
def generate_images(model, noise, class_label):
    model.eval()
    with torch.no_grad():
        for _ in range(model.num_steps):
            noise = model(noise, class_label)
        recon_image = noise.clone()
    return recon_image

# ---------------------------
# 5) 학습 루프 (원문 유지)
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
patience = max_patience

for epoch in range(max_epochs):
    total_train_loss = {"loss_class": 0, "loss_pixel": 0, "loss_total": 0}
    total_val_loss   = {"loss_class": 0, "loss_pixel": 0, "loss_total": 0}

    model.train()

    # training
    for iter, (image, label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device)

        # Forward diffusion process
        noisy_images = [image.clone()]
        noisy_image = image.clone()
        for _ in range(model.num_steps):
            noisy_image = model(noisy_image, backward=False)
            noisy_images.append(noisy_image.clone())

        optimizer.zero_grad()

        # Backward diffusion process
        recon_image = noisy_images.pop()
        train_pixel_loss = torch.tensor(0.).to(device)
        train_class_loss = torch.tensor(0.).to(device)

        for _ in range(model.num_steps):
            recon_image = model(recon_image, label)
            train_pixel_loss += pixelwise_loss(recon_image, image)
            class_pred = c_model(recon_image)
            train_class_loss += classification_loss(class_pred, label)
            recon_image = noisy_images.pop()

        this_train_loss = train_pixel_loss + 0.025 * train_class_loss
        this_train_loss.backward()
        optimizer.step()

        total_train_loss["loss_class"] += train_class_loss.item()
        total_train_loss["loss_pixel"] += train_pixel_loss.item()
        total_train_loss["loss_total"] += this_train_loss.item()

    scheduler.step()
    total_train_loss = {k: v / (iter + 1) for k, v in total_train_loss.items()}

    # validation
    model.eval()

    for iter, (image, label) in enumerate(val_dataloader):
        image = image.to(device)
        label = label.to(device)

        # Forward diffusion process
        noisy_images = [image.clone()]
        noisy_image = image.clone()
        for _ in range(model.num_steps):
            noisy_image = model(noisy_image, backward=False)
            noisy_images.append(noisy_image.clone())

        # Backward diffusion process
        recon_image = noisy_images.pop()
        val_pixel_loss = torch.tensor(0.).to(device)
        val_class_loss = torch.tensor(0.).to(device)

        for _ in range(model.num_steps):
            recon_image = model(recon_image, label)
            val_pixel_loss += pixelwise_loss(recon_image, image)
            class_pred = c_model(recon_image)
            val_class_loss += classification_loss(class_pred, label)
            _ = noisy_images.pop()

        total_val_loss["loss_class"] += val_class_loss.item()
        total_val_loss["loss_pixel"] += val_pixel_loss.item()
        total_val_loss["loss_total"] += (val_pixel_loss + 0.025 * val_class_loss).item()

    total_val_loss = {k: v / (iter + 1) for k, v in total_val_loss.items()}

    print(f"\nEpoch: {epoch+1}/{max_epochs},"
          f"\n\tTrain Loss: {total_train_loss['loss_total']:.8f}"
          f"\n\t\tclassification loss: {total_train_loss['loss_class']:.8f}"
          f"\n\t\tpixelwise loss: {total_train_loss['loss_pixel']:.8f}"
          f"\n\tVal Loss: {total_val_loss['loss_total']:.8f}"
          f"\n\t\tclassification loss: {total_val_loss['loss_class']:.8f}"
          f"\n\t\tpixelwise loss: {total_val_loss['loss_pixel']:.8f}")

    noise = torch.randn(1, 1, 28, 28).repeat(10, 1, 1, 1).to(device)
    class_label = torch.tensor([n for n in range(10)]).to(device)
    result = generate_images(model, noise, class_label).cpu().detach().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=10)
    for i, img in enumerate(result):
        plt.subplot(1, 10, i+1)
        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()
    plt.clf()

    if total_val_loss["loss_total"] < best_val_loss:
        best_val_loss = total_val_loss["loss_total"]
        print(f"\tSaving the model state dictionary for Epoch: {epoch+1} with Validation loss: {best_val_loss:.8f}")
        torch.save(model.state_dict(), "mnist_diffusion.dth")
        patience = max_patience
    else:
        patience -= 1
        print(f"\tLoss not decreased. Will wait for {patience} more epochs...")

    if patience <= 0:
        break

print("\n[INFO] Training done. Best model saved as mnist_diffusion.dth")
