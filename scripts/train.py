"""
Train a CIFAR-10 classifier for the Section 3 hyperparameter sweep.

Single-run script: one architecture / optimizer / lr / wd / seed combination.
For the full paper sweep, wrap this in a shell loop or a job array.

Example (single run):
    python scripts/train.py --arch ResNet18 --optimizer SGD --lr 0.1 \
        --weight_decay 5e-4 --seed 0 --epochs 200 \
        --data_dir /path/to/cifar10 --checkpoint_dir ./checkpoints/ResNet18

Example (mini sweep via shell):
    for arch in ResNet18 VGG19 MobileNet; do
      for lr in 0.01 0.1 1.0; do
        python scripts/train.py --arch $arch --optimizer SGD --lr $lr \
            --weight_decay 5e-4 --seed 0 --data_dir /data/cifar10 \
            --checkpoint_dir ./checkpoints/${arch}
      done
    done

Paper sweep parameters (Section 3):
    --arch: ResNet18, ResNet34, ResNet50, VGG13, VGG19, DenseNet121, MobileNet, EfficientNetB0
    --optimizer: SGD, AdamW
    --lr: 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0  (varies by arch)
    --weight_decay: 0, 2e-4, 5e-4, 1e-3
    --seed: 0, 1, 2
    --epochs: 200
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model, ARCH_NAMES


# ── Data ──────────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_loaders(data_dir, batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def ckpt_filename(epoch, args):
    return (f"ckpt_epoch={epoch:04d}"
            f"_arch={args.arch}"
            f"_opt={args.optimizer}"
            f"_lr={args.lr}"
            f"_wd={args.weight_decay}"
            f"_seed={args.seed}.pt")


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def find_latest_checkpoint(checkpoint_dir, args):
    """Return (epoch, path) of the most recent checkpoint, or (0, None)."""
    if not os.path.isdir(checkpoint_dir):
        return 0, None
    prefix = (f"_arch={args.arch}_opt={args.optimizer}"
              f"_lr={args.lr}_wd={args.weight_decay}_seed={args.seed}.pt")
    candidates = [f for f in os.listdir(checkpoint_dir)
                  if f.startswith("ckpt_epoch=") and f.endswith(prefix)]
    if not candidates:
        return 0, None
    latest = max(candidates, key=lambda f: int(f.split("=")[1].split("_")[0]))
    epoch = int(latest.split("=")[1].split("_")[0])
    return epoch, os.path.join(checkpoint_dir, latest)


# ── Train / eval ──────────────────────────────────────────────────────────────

def run_epoch(net, loader, criterion, optimizer, device, train=True):
    net.train() if train else net.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CIFAR-10 model (Section 3 sweep)")
    p.add_argument('--arch',         default='ResNet18',  choices=ARCH_NAMES)
    p.add_argument('--optimizer',    default='SGD',       choices=['SGD', 'AdamW'])
    p.add_argument('--lr',           default=0.1,         type=float)
    p.add_argument('--weight_decay', default=5e-4,        type=float)
    p.add_argument('--momentum',     default=0.9,         type=float)
    p.add_argument('--epochs',       default=200,         type=int)
    p.add_argument('--batch_size',   default=128,         type=int)
    p.add_argument('--seed',         default=0,           type=int)
    p.add_argument('--data_dir',     default='./data/cifar10')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--save_all',     action='store_true',
                   help='Save a checkpoint every epoch (default: save only final epoch)')
    p.add_argument('--resume',       action='store_true',
                   help='Resume from latest checkpoint in checkpoint_dir')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: arch={args.arch} opt={args.optimizer} lr={args.lr} "
          f"wd={args.weight_decay} seed={args.seed} epochs={args.epochs}")

    trainloader, testloader = get_loaders(args.data_dir, args.batch_size)

    net = build_model(args.arch, dataset='cifar').to(device)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        last_epoch, ckpt_path = find_latest_checkpoint(args.checkpoint_dir, args)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt['model']
            if all(k.startswith('module.') for k in state_dict):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            base = net.module if isinstance(net, nn.DataParallel) else net
            base.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = last_epoch + 1
            print(f"Resumed from epoch {last_epoch}, acc={ckpt['acc']:.2f}%")
        else:
            print("No checkpoint found; starting from scratch.")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = run_epoch(net, trainloader, criterion, optimizer, device, train=True)
        test_loss,  test_acc  = run_epoch(net, testloader,  criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:03d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
              f"test loss {test_loss:.4f} acc {test_acc:.2f}%", flush=True)

        if args.save_all or epoch == args.epochs - 1:
            base = net.module if isinstance(net, nn.DataParallel) else net
            state = {
                'epoch':     epoch,
                'model':     base.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc':       test_acc,
                'params': {
                    'arch':         args.arch,
                    'optimizer':    args.optimizer,
                    'lr':           args.lr,
                    'weight_decay': args.weight_decay,
                    'seed':         args.seed,
                    'epochs':       args.epochs,
                },
            }
            path = os.path.join(args.checkpoint_dir, ckpt_filename(epoch, args))
            save_checkpoint(state, path)
            print(f"  Saved checkpoint: {path}", flush=True)

    print("Training complete.")


if __name__ == '__main__':
    main()
