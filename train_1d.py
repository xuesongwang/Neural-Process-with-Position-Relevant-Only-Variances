import argparse
import time
from tqdm import tqdm
import numpy as np
import torch
import stheno.torch as stheno

from data.GP_data import GPGenerator
from data.smart_meter import SmartMeterDataLoader
from model.utils import gaussian_logpdf, RunningAverage
from model.NP_Prov import UNet, NPPROV


def train(data, model, opt):
    """Perform a training epoch."""
    ravg = RunningAverage()
    model.train()
    total_recover_loss = 0
    for step, task in enumerate(data):
        y_mean, y_std, recover_loss = model(task['x_context'].to(device), task['y_context'].to(device), task['x_target'].to(device))
        nll = -gaussian_logpdf(task['y_target'].to(device), y_mean, y_std, 'batched_mean')
        obj = recover_loss + nll
        obj.backward()
        opt.step()
        opt.zero_grad()
        ravg.update(nll.item() / data.batch_size, data.batch_size)
        total_recover_loss += recover_loss.item()
    return ravg.avg, total_recover_loss/(step+1)

def validation(data, model):
    """Perform a training epoch."""
    ravg = RunningAverage()
    model.eval()
    total_recover_loss = 0
    for step, task in enumerate(data):
        y_mean, y_std, recover_loss = model(task['x_context'].to(device), task['y_context'].to(device), task['x_target'].to(device))
        nll = -gaussian_logpdf(task['y_target'].to(device), y_mean, y_std, 'batched_mean')
        ravg.update(nll.item() / data.batch_size, data.batch_size)
        total_recover_loss += recover_loss.item()
    return ravg.avg, total_recover_loss/(step+1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        default='smart_meter',
                        type=str,
                        choices=['EQ',
                                 'matern',
                                 'period',
                                 'smart_meter'],
                        help='Data set to train the NP-PROV on. ')

    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        help='Number of epochs to train for.')
    parser.add_argument('--learning_rate',
                        default=3e-4,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--weight_decay',
                        default=1e-5,
                        type=float,
                        help='Weight decay.')
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Generate a model
    t0 = time.time()
    model = NPPROV(rho=UNet(), points_per_unit=64)
    model.to(device)
    print("model to device time: ", time.time() - t0)

    # Build data generator
    if args.name == 'EQ':
        kernel = stheno.EQ().stretch(0.25)
        gen = GPGenerator(kernel=kernel)
        gen_val = GPGenerator(kernel=kernel, num_tasks=60)
    elif args.name == 'matern':
        kernel = stheno.Matern52().stretch(0.25)
        gen = GPGenerator(kernel=kernel)
        gen_val = GPGenerator(kernel=kernel, num_tasks=60)
    elif args.name == 'period':
        kernel = stheno.EQ().stretch(0.5) * stheno.EQ().periodic(period=0.25)
        gen = GPGenerator(kernel=kernel)
        gen_val = GPGenerator(kernel=kernel, num_tasks=60)
    elif args.name == 'smart_meter':
        dataloader = SmartMeterDataLoader()
        gen = dataloader.train_dataloader()
        gen_eval = dataloader.val_dataloader()



    PLOT_FREQ = 10
    # Initialize optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    best_obj = np.inf
    # Run the training loop.
    for epoch in tqdm(range(args.epochs)):
        # Compute training objective.
        train_obj, mse = train(gen, model, opt)
        val_obj, mse_val = validation(gen_val, model)

        # Update the best objective value and checkpoint the model.
        if val_obj < best_obj:
            best_obj = val_obj
            print("save model at epoch: %d, val log likelihood: %.3f, mse loss: %.3f" % (epoch, -best_obj, mse_val))
            # save model
            # torch.save(model.state_dict(), 'saved_model/' + args.name + '_model.pt')

        # Plot model behaviour every now and again.
        if epoch % PLOT_FREQ == 0:
            print('Epoch %s: training log likelihood %.3f' % (epoch, -train_obj))




