from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN, ImageFolder
import torchvision.transforms as tf
import numpy as np

def train_val_split(trainset):
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(0.3 * num_train))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler

def load_dataset(name, batchsize):
    if name == 'mnist':
        trainset = MNIST('./MNIST/mnist_data', train=True, download=False,
                     transform=tf.ToTensor())
        testset = MNIST('./MNIST/mnist_data', train=False, download=False,
                        transform=tf.ToTensor())
        trainsampler, validsampler = train_val_split(trainset)
        trainloader = DataLoader(trainset, batch_size=batchsize, sampler=trainsampler, shuffle=False, num_workers=8)
        valloader = DataLoader(trainset, batch_size=batchsize, sampler=validsampler, shuffle=False)
        testloader = DataLoader(testset, batch_size=batchsize, shuffle=True)
    elif name == 'svhn':
        trainset = SVHN('./SVHN', split='train', download=False,
                        transform=tf.ToTensor())
        testset = SVHN('./SVHN', split='test', download=False,
                       transform=tf.ToTensor())
        trainsampler, validsampler = train_val_split(trainset)
        trainloader = DataLoader(trainset, batch_size=batchsize, sampler=trainsampler, shuffle=False, num_workers=8)
        valloader = DataLoader(trainset, batch_size=batchsize, sampler=validsampler, shuffle=False)
        testloader = DataLoader(testset, batch_size=batchsize, shuffle=True)
    elif name == 'celebA':
        transform = tf.Compose([
            tf.Resize([32, 32]),
            tf.ToTensor(),
        ])
        trainset = ImageFolder('./celebA/train/', transform)
        trainsampler, validsampler = train_val_split(trainset)
        trainloader = DataLoader(trainset, batch_size=batchsize, sampler=trainsampler, shuffle=False, num_workers=8, drop_last=True)
        valloader = DataLoader(trainset, batch_size=batchsize, sampler=validsampler, shuffle=False, drop_last=True)
        testset = ImageFolder('./celebA/test/', transform)
        testloader = DataLoader(dataset=testset, batch_size=batchsize, shuffle=False, num_workers=8,
                                drop_last=True)
    return trainloader, valloader, testloader
