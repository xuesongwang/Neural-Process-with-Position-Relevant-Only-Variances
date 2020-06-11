import argparse
from tqdm import tqdm
import torch
from data.image_data import load_dataset
from model.NP_Prov_2d import NPPROV2d
from model.utils import channel_last
import warnings

warnings.filterwarnings("ignore")


def validate(model, dataloader):
    model.eval()
    avg_loss = 0
    mse_loss = 0
    for index, (I, _) in tqdm(enumerate(dataloader)):
        I = I.to(args.device)
        pred_dist, mse, _ = model(I)
        loss = - pred_dist.log_prob(channel_last((I))).mean()
        avg_loss += loss.item()
        mse_loss += mse.item()
    return -avg_loss/(index+1),  mse_loss / (index + 1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-B', type=int, default=16)
    parser.add_argument('--dataset', '-D', type=str, default='mnist', choices=['mnist', 'svhn', 'celebA'])

    args = parser.parse_args()
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if args.dataset == 'mnist':
        _, _,  testloader = load_dataset(args.dataset, args.batch_size)
        npprov = NPPROV2d(channel=1)
    elif args.dataset == 'svhn':
        _, _,  testloader = load_dataset(args.dataset, args.batch_size)
        npprov = NPPROV2d(channel=3)
    elif args.dataset == 'celebA':
        _, _,  testloader = load_dataset(args.dataset, args.batch_size)
        npprov = NPPROV2d(channel=3)

    npprov = npprov.to(args.device)
    filename = 'saved_model/{}_model.pth.gz'.format(args.dataset)
    npprov.load_state_dict(torch.load(filename))
    print("load model success!")

    valid_ll, mse_loss_val = validate(npprov, testloader)
    print("testing log likelihood: %.4f, mse:  %.4f" % (valid_ll, mse_loss_val))