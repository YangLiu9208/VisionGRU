import math
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from dataset import Imagenet, get_transform
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch_ema import ExponentialMovingAverage
from timm.data import Mixup
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def try_get_from_env(name, default):
    if name in os.environ:
        env_value = os.environ.get(name)
        try:
            if isinstance(default, bool):
                if env_value.lower() in ['true', '1']:
                    env_value =  True
                elif env_value.lower() in ['false', '0']:
                    env_value =  False
                else:
                    env_value =  default
            else:
                env_value = type(default)(env_value)
        except (ValueError, TypeError):
            print(f'bad value {env_value}',end=' ')
            env_value = default
    else:
        env_value = default
    print(f"use {name}: {env_value}")
    return env_value

def initialize_datasets():
    data_root = try_get_from_env('data_dir', 'data')
    train_dataset = Imagenet(root_dir=f'{data_root}/ILSVRC/Data/CLS-LOC/train', transform=get_transform(), full=True)
    return train_dataset, None

num_cards = try_get_from_env('NC', 1)
train_dataset, validation_dataset = initialize_datasets()
num_classes = len(train_dataset.label_to_index)
batch_size = try_get_from_env('B', 256) 
batch_size //= num_cards

def init_distributed():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    return torch.device(f'cuda:{dist.get_rank()}')

device = init_distributed()

from visionGRU import initialize_model, num_epochs

ckpt = 'vig'
print(ckpt)
ckpt_path = 'ckpt'
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
restart_epoch = try_get_from_env("RE", 0)
checkpoint_file = f"ckpt/{ckpt}_{restart_epoch}_model_path.pth"
model = initialize_model(num_classes, compile=try_get_from_env('COMPILE', False)).to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
if restart_epoch:
    model.module.load_state_dict(torch.load(checkpoint_file, map_location=map_location))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3 * batch_size * num_cards / 1024,
    betas=(0.9, 0.999),
    weight_decay=0.05
)
opt_checkpoint_file = checkpoint_file.replace('model', 'optimizer')
if restart_epoch:
    optimizer.load_state_dict(torch.load(opt_checkpoint_file))

steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * num_cards))
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_epochs//15 * steps_per_epoch,
    num_training_steps=num_epochs * steps_per_epoch
)
if restart_epoch:
    for _ in range(restart_epoch*steps_per_epoch):
        scheduler.step()

mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    label_smoothing=0.1,
    num_classes=num_classes
)
ema_checkpoint_file = checkpoint_file.replace('model', 'ema')
ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
if restart_epoch:
    ema.load_state_dict(torch.load(ema_checkpoint_file))

def log(op, idx):
    if dist.get_rank() == 0:
        print(op)
        with open(f'{idx}.log', 'a') as f:
            f.write(op + '\n')

use_amp = try_get_from_env('AMP', False)
def train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs):
    if use_amp:
        scaler = GradScaler()
    for epoch in range(restart_epoch, num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        train_sampler.set_epoch(epoch)
        for _, (images, labels) in tqdm(enumerate(train_loader), disable=dist.get_rank() != 0):
            images, labels = images.to(device), labels.to(device)
            if images.size(0) % 2 != 0:
                images, labels = images[:-1], labels[:-1]
            images, labels = mixup_fn(images, labels)
            optimizer.zero_grad()
            if use_amp:
                with autocast():
                    outputs = model(images)
                    log_probs = torch.log_softmax(outputs, dim=1)
                    loss = criterion(log_probs, labels.to(log_probs.device))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                log_probs = torch.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, labels.to(log_probs.device)) 
                loss.backward()
                optimizer.step()

            scheduler.step()
            ema.update()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.max(1)[1]).sum().item()
        with torch.no_grad():
            running_loss_tensor = torch.tensor(running_loss, dtype=torch.float32, device=device)
            correct_tensor = torch.tensor(correct, dtype=torch.float32, device=device)
            total_tensor = torch.tensor(total, dtype=torch.float32, device=device)

            dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

            train_loss = running_loss / total
            train_accuracy = 100 * correct / total
            if epoch % 5 == 0 or epoch > 100:
                if dist.get_rank() == 0:
                    torch.save(optimizer.state_dict(), f'{ckpt_path}/{ckpt}_{epoch}_optimizer_path.pth')
                    torch.save(model.module.state_dict(), f'{ckpt_path}/{ckpt}_{epoch}_model_path.pth')
                    torch.save(ema.state_dict(), f'{ckpt_path}/{ckpt}_{epoch}_ema_path.pth')
            log(f'Epoch {epoch + 1}/{num_epochs}, Loss {train_loss}, TrainAcc: {train_accuracy:.2f}%', ckpt)

if __name__ == '__main__':
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=24, pin_memory=True)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    if dist.get_rank() == 0:
        print('train start')

    train_model(model, train_loader, criterion, optimizer)
