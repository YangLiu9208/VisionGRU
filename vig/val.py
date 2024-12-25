import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch_ema import ExponentialMovingAverage
from dataset import ValidationDataset, get_transform, Imagenet
from visionGRU import initialize_model
import torch.nn.functional as F
from tqdm import tqdm

os.environ["TORCHINDUCTOR_DISABLE"] = "1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def init_distributed():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    return torch.device(f'cuda:{dist.get_rank()}')


device = init_distributed()
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

if dist.get_rank() == 0:
    print('loading dataset')
data_root = try_get_from_env('data_dir', 'data')
train_dataset = Imagenet(root_dir=f'{data_root}/ILSVRC/Data/CLS-LOC/train', transform=get_transform(), full=True)
validation_dataset = ValidationDataset(
    root_dir=f'{data_root}/ILSVRC/Data/CLS-LOC/val',
    annotations_dir=f'{data_root}/ILSVRC/Annotations/CLS-LOC/val',
    label_to_index=train_dataset.label_to_index,
    transform=get_transform(False)
)
def log(op, idx):
    if dist.get_rank() == 0:  
        print(op)
        with open(f'{idx}.log', 'a') as f:
            f.write(op + '\n')


val_sampler = DistributedSampler(validation_dataset, shuffle=False)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=try_get_from_env("B", 64),
    sampler=val_sampler,
    num_workers=24,
    pin_memory=True
)
num_classes = len(validation_dataset.label_to_index)
model = initialize_model(num_classes, compile=False).to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])


def evaluate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for idx, (images, labels) in tqdm(enumerate(dataloader), disable=dist.get_rank()!=0):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            outputs = torch.log_softmax(outputs, dim=1)
            loss = criterion(outputs, labels_one_hot)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    correct_tensor = torch.tensor(correct, dtype=torch.float32, device=device)
    total_tensor = torch.tensor(total, dtype=torch.float32, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    accuracy = 100 * correct_tensor.item() / total_tensor.item()
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

if __name__ == '__main__':
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    if dist.get_rank() == 0:
        name = try_get_from_env("NAME", f'{time.time()}')
        print('Validation start')
    model_t = initialize_model(num_classes)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}

    ckpt = try_get_from_env("ckpt",'')
    state_dict = torch.load(ckpt, map_location=map_location)
    new_dict = {}
    for key in state_dict:
        new_dict[key.removeprefix("_orig_mod.")] = state_dict[key]

    map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
    model.module.load_state_dict(new_dict)

    accuracy, avg_loss = evaluate(model, validation_loader, criterion)

    if dist.get_rank() == 0:
        log(
            f'Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_loss:.4f}', name)