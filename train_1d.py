"""
this version I try to change attention weight into convolution
"""

import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import stheno.torch as stheno

from model.data import GPGenerator, SawtoothGenerator, GPGeneratorNew
from utils import gaussian_logpdf, RunningAverage
from model.architectures_matrix_prov import UNet, ConvCNP

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()

def train(data, model, opt):
    """Perform a training epoch."""
    ravg = RunningAverage()
    model.train()
    total_recover_loss = 0
    for step, task in enumerate(data):
        y_mean, y_std, recover_loss = model(task['x_context'], task['y_context'], task['x_target'])
        nll = -gaussian_logpdf(task['y_target'], y_mean, y_std, 'batched_mean')
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
        y_mean, y_std, recover_loss, original_min, original_max = model(task['x_context'], task['y_context'], task['x_target'], normalize = 'minmax')
        y_mean = (y_mean * (original_max - original_min) + original_min)
        nll = -gaussian_logpdf(task['y_target'], y_mean, y_std, 'batched_mean')
        obj = recover_loss + nll
        ravg.update(nll.item() / data.batch_size, data.batch_size)
        total_recover_loss += recover_loss.item()
    return ravg.avg, total_recover_loss/(step+1)

def plot_model_task(model, task, idx, legend):
    # Create a set of target points corresponding to entire [-pi, pi] range
    x_target_index = torch.Tensor(np.linspace(-2, 2, 100)).to(device)
    x_target_index = x_target_index.unsqueeze(1).unsqueeze(0).to(device)
    # Make predictions with the model.
    model.eval()
    batch_size = task['x_context'].size(0)
    with torch.no_grad():
        y_mean, y_std = model(task['x_context'], task['y_context'], x_target_index.repeat(batch_size,1,1))

    # Plot the task and the model predictions.
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx])
    y_mean, y_std = to_numpy(y_mean[idx]), to_numpy(y_std[idx])

    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set', color='black')
    plt.scatter(x_target, y_target, label='Target Set', color='red')

    # Plot model predictions.
    plt.plot(to_numpy(x_target_index[0]), y_mean, label='Model Output', color='blue')
    plt.fill_between(to_numpy(x_target_index[0]),
                     y_mean + 2 * y_std,
                     y_mean - 2 * y_std,
                     color='tab:blue', alpha=0.2)
    if legend:
        plt.legend()

t0 = time.time()
model = ConvCNP(rho=UNet(), points_per_unit=64)
model.to(device)
print("model to device", time.time() - t0)


name = 'EQ'
# name = 'period'
# name = 'matern'
# name = 'Sawtooth'
if name == 'EQ':
    kernel = stheno.EQ().stretch(0.25)
elif name == 'matern':
    kernel = stheno.Matern52().stretch(0.25)
elif name == 'period':
    kernel = stheno.EQ().stretch(0.5) * stheno.EQ().periodic(period=0.25)
else:
    gen = SawtoothGenerator()
    gen_val = SawtoothGenerator(num_tasks=60)
# gen = GPGeneratorNew(kernel=kernel)
# gen_val = GPGeneratorNew(kernel=kernel, num_tasks=60)
gen = GPGenerator(kernel=kernel)
gen_val = GPGenerator(kernel=kernel, num_tasks=60)

# Some training hyper-parameters:
LEARNING_RATE = 3e-4 if name != 'Sawtooth' else 3e-4
NUM_EPOCHS = 200
PLOT_FREQ = 10

# Initialize optimizer
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

best_obj = np.inf
# Run the training loop.
for epoch in tqdm(range(NUM_EPOCHS)):
    # Compute training objective.
    train_obj, mse = train(gen, model, opt)
    val_obj, mse_val = validation(gen_val, model)

    # Update the best objective value and checkpoint the model.
    if val_obj < best_obj:
        best_obj = val_obj
        print("save model at epoch: %d, val log likelihood: %.3f, mse loss: %.3f" % (epoch, -best_obj, mse_val))
        # save model
        torch.save(model.state_dict(), 'saved_model/' + name + '/Ours/model_matrix_minmaxscaled.pt')

    # Plot model behaviour every now and again.
    if epoch % PLOT_FREQ == 0:
        print('Epoch %s: LogLikelihood %.3f' % (epoch, -train_obj))
    #     task = gen.generate_task()
    #     fig = plt.figure(figsize=(24, 5))
    #     task = gen.generate_task()
    #     for i in range(3):
    #         plt.subplot(1, 3, i + 1)
    #         plot_model_task(model, task, idx=i, legend=i==2)
    #     plt.show()


