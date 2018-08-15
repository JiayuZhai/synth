import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
from model import *
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-model',default='GAN',help="The GAN model we used. Default is WGAN-GP.")
parser.add_argument('-hs',default=50,help="Hidden unit number for both D and G.")
parser.add_argument('-visual',default=True,help='Use visdom to show visual result or not. Default is True.')
args = parser.parse_args()
arg_dict = vars(args)

visual = bool(arg_dict['visual'])
if visual:
    import visdom
    vis = visdom.Visdom()

# Model params
g_input_size = 100     # Random noise dimension coming into generator, per output vector
g_hidden_size = int(arg_dict['hs'])   # Generator complexity

d_hidden_size = g_hidden_size   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
minibatch_size = 100

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 10000
print_interval = 200
d_steps = 5  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

MODEL =  arg_dict['model'] # 'GAN', 'GAN_Gumbel_softmax', 'WGAN_GP'
use_cuda = False
LAMBDA = 10 # Gradient penalty lambda hyperparameter
STAT = True # show real data distribution and synthetic data distribution or not

table = np.load('ICEM_preprocessed_new.npy')
ppp = PreprocessPCAPostprocess(table, [True,True,False,True,True,True,True,True,False,False])
ppp.preprocess()
ppp.pca_fit()
table = ppp.pca_transform()
g_output_size = table.shape[1]    # size of generated output vector
d_input_size = table.shape[1]   # Minibatch size - cardinality of distributions
np.random.shuffle(table)
var_distinct = np.load('categories.npy')


# Function of WGAN-GP's GP
def calc_gradient_penalty(D, real_data, fake_data):
    alpha = torch.rand(minibatch_size, 1)
    alpha = alpha.expand(minibatch_size, real_data.nelement()/minibatch_size).contiguous().view(minibatch_size, -1)
    alpha = alpha.cuda(gpu) if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty    

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)

# Create sampler object for sampling real data and noise
sampler = Sampler(table, minibatch_size)
d_sampler = sampler.get_distribution_sampler()
gi_sampler = sampler.get_generator_input_sampler()

# Create Generator and Discriminator object
if MODEL == 'WGAN_GP':
    G = Generator_WGANGP_basic(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    D = Discriminator_WGANGP(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
else:
    if MODEL == 'GAN_Gumbel_softmax':
        G = Generator_Gumbel_softmax(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    else:
        G = Generator_basic(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)
    criterion = nn.BCELoss()

print('Generator architecture', G)
print('Discriminator architecture', D)
# Create optimizer by only Generator or only Discriminator parameters
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

# Use for Visualisation and Training
if MODEL == 'WGAN_GP':
    Wasserstein_per_epoch = []
    one = torch.FloatTensor([1])
    mone = one * -1

D_real_error_per_epoch = []
D_fake_error_per_epoch = []
G_error_per_epoch = []

# TRAINING LOOP
for epoch in tqdm(range(num_epochs)):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(d_real_data)
        if MODEL == 'WGAN_GP':
            d_real_error = d_real_decision.mean()
            d_real_error.backward(mone)
        else:
            d_real_error = criterion(d_real_decision, Variable(torch.ones(minibatch_size)))  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params
        
        #  1B: Train D on fake
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(d_fake_data)
        if MODEL == 'WGAN_GP':
            d_fake_error = d_fake_decision.mean()
            d_fake_error.backward(one)
        else:
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(minibatch_size)))  # zeros = fake
            d_fake_error.backward()
        
        #  1C: Train D on gradient penalty
        if MODEL == 'WGAN_GP':
            gradient_penalty = calc_gradient_penalty(D, d_real_data, d_fake_data)
            gradient_penalty.backward()
            
            Wasserstein_per_epoch.append(d_real_error - d_fake_error)
        
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        g_fake_data = G(gen_input)

        dg_fake_decision = D(g_fake_data)
        if MODEL == 'WGAN_GP':
            g_error = dg_fake_decision.mean()
            g_error.backward(mone)
        else:
            g_error = criterion(dg_fake_decision, Variable(torch.ones(minibatch_size)))  # we want to fool, so pretend it's all genuine
            g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters

    if epoch % print_interval == 0:

        D_real_error_per_epoch.append(extract(d_real_error)[0])
        D_fake_error_per_epoch.append(extract(d_fake_error)[0])
        G_error_per_epoch.append(extract(g_error)[0])


torch.save(D.state_dict(),MODEL+'/D_model.model')
torch.save(G.state_dict(),MODEL+'/G_model.model')


import matplotlib.pyplot as plt

plt.figure()
plt.plot(D_real_error_per_epoch,'b')
plt.plot(D_fake_error_per_epoch,'r')
plt.plot(G_error_per_epoch,'y')
if visual:
    vis.matplot(plt)
plt.savefig(MODEL+'/error1.pdf',format='pdf')
if MODEL == 'WGAN_GP':
    plt.figure()
    plt.plot(Wasserstein_per_epoch)
    if visual:
        vis.matplot(plt)
    plt.savefig('Wasserstein.pdf',format='pdf')
# plt.show()