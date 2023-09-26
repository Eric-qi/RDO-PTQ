import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder



def get_dataloader(config):
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(config['patchsize']), transforms.ToTensor()]
    )
    train_dataset = ImageFolder(config['trainset'], split=config['c_data'], transform=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batchsize'],
        num_workers=config['worker_num'],
        shuffle=True,
        pin_memory=True,
    )


    test_transforms = transforms.Compose(
        [transforms.CenterCrop(config['patchsize']), transforms.ToTensor()]
    )
    test_dataset = ImageFolder(config['trainset'], split=config['t_data'], transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batchsize_test'],
        num_workers=config['worker_num'],
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader

def get_train_samples_imagenet(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples] # output 3-dim tensor


def get_train_samples(train_loader, num_samples):
    train_data = None
    for i, batch in enumerate(train_loader):
        if train_data is None:
            train_data = batch
        else:
            train_data = torch.cat((train_data, batch))
        
        if train_data.shape[0] >= num_samples:
            break
    return train_data[:num_samples] # output 4-dim tensor for LIC