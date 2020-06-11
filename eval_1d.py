import argparse
import time
from tqdm import tqdm
import numpy as np
import torch
import stheno.torch as stheno
from data.smart_meter import SmartMeterDataLoader
from data.GP_data import GPGenerator
from model.utils import gaussian_logpdf, RunningAverage
from model.NP_Prov import UNet, NPPROV


def validation(data, model):
    """Perform a training epoch."""
    ravg = RunningAverage()
    model.eval()
    total_recover_loss = 0
    for step, task in enumerate(data):
        if step % 100 == 99:
            print("current step", step+1)
        y_mean, y_std, recover_loss = model(task['x_context'].to(device), task['y_context'].to(device), task['x_target'].to(device))
        nll = gaussian_logpdf(task['y_target'].to(device), y_mean, y_std, 'batched_mean')
        ravg.update(nll.item() / data.batch_size, data.batch_size)
        total_recover_loss += recover_loss.item()
    return ravg.avg, total_recover_loss/(step+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        default='EQ',
                        type=str,
                        choices=['EQ',
                                 'matern',
                                 'period',
                                 'smart_meter'],
                        help='Data set to train the NP-PROV on. ')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Generate a model
    t0 = time.time()
    model = NPPROV(rho=UNet(), points_per_unit=64)
    model.load_state_dict(torch.load("./saved_model/" + args.name + "_model.pt", map_location="cuda:0" if torch.cuda.is_available() else 'cpu'))

    model.to(device)
    print("load model success! model to device time: ", time.time() - t0)

    # Build data generator
    if args.name == 'EQ':
        kernel = stheno.EQ().stretch(0.25)
        gen_test = GPGenerator(kernel=kernel, num_tasks=2048)
    elif args.name == 'matern':
        kernel = stheno.Matern52().stretch(0.25)
        gen_test = GPGenerator(kernel=kernel, num_tasks=2048)
    elif args.name == 'period':
        kernel = stheno.EQ().stretch(0.5) * stheno.EQ().periodic(period=0.25)
        gen_test = GPGenerator(kernel=kernel, num_tasks=2048)
    elif args.name == 'smart_meter':
        dataloader = SmartMeterDataLoader()
        gen_test = dataloader.test_dataloader()


    ll_test, mse_test = validation(gen_test, model)
    print("test log likelihood: %.3f, mse loss: %.3f" % (ll_test, mse_test))




