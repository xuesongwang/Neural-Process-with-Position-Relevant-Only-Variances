import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from data.image_data import load_dataset
from model.NP_Prov_2d import NPPROV2d
from model.utils import channel_last
import warnings

warnings.filterwarnings("ignore")

def train(model, dataloader, optimizer):
    model.train()
    avg_loss = 0
    mse_loss = 0
    for index, (I, _) in tqdm(enumerate(dataloader)):
        I = I.to(args.device)

        optimizer.zero_grad()

        pred_dist, mse,_ = model(I)
        nll = - pred_dist.log_prob(channel_last((I))).mean()
        loss = nll + mse

        loss.backward()
        optimizer.step()

        avg_loss += nll.item()
        mse_loss += mse.item()
    return -avg_loss / (index+1), mse_loss/(index +1)

def validate(model, dataloader):
    model.eval()
    avg_loss = 0
    mse_loss = 0
    for index, (I, _) in enumerate(dataloader):
        # if index % 100 == 99:
        #     print(index +1)
        I = I.to(args.device)
        pred_dist, mse, _ = model(I)
        loss = - pred_dist.log_prob(channel_last((I))).mean()
        avg_loss += loss.item()
        mse_loss += mse.item()
    return -avg_loss/(index+1),  mse_loss / (index + 1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-B', type=int, default=16)
    parser.add_argument('--learning-rate', '-LR', type=float, default=5e-4)
    parser.add_argument('--epochs', '-E', type=int, default=100)
    parser.add_argument('--dataset', '-D', type=str, default='celebA', choices=['mnist', 'svhn', 'celebA'])

    args = parser.parse_args()
    filename = 'saved_model/{}_model.pth.gz'.format(args.dataset)
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if args.dataset == 'mnist':
        trainloader, valloader, _ = load_dataset(args.dataset, args.batch_size)
        npprov = NPPROV2d(channel=1)
    elif args.dataset == 'svhn':
        trainloader, valloader, _ = load_dataset(args.dataset, args.batch_size)
        npprov = NPPROV2d(channel=3)
    elif args.dataset == 'celebA':
        trainloader, valloader, _ = load_dataset(args.dataset, args.batch_size)
        npprov = NPPROV2d(channel=3)

    npprov = npprov.to(args.device)
    optimizer = optim.Adam(npprov.parameters(), lr=args.learning_rate)

    best_ll = -np.inf
    patience = 0
    MAX_PATIENCE = 15
    for epoch in range(args.epochs):
        avg_train_loss, mse_loss= train(npprov, trainloader, optimizer)
        valid_ll, mse_loss_val = validate(npprov, valloader)
        print("epoch: %d, validation log likelihood: %.4f, %.4f" % (epoch, valid_ll, mse_loss_val))
        if valid_ll > best_ll:
            patience = 0
            best_ll = valid_ll
            print("saved model at epocch: %d, ll: %.3f, mse: %.3f"%(epoch, avg_train_loss, mse_loss))
            # torch.save(npprov.state_dict(), filename)
        else:
            patience += 1
        if patience == MAX_PATIENCE:
            print("early stopping at epoch:", epoch)
            break