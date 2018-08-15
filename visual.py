import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from model import *
import argparse
from collections import defaultdict
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-model',default='WGAN_GP',help="The GAN model we used. Default is WGAN-GP.")
parser.add_argument('-hs',default=50,help="Hidden unit number for both D and G.")
args = parser.parse_args()
arg_dict = vars(args)

# Model params
g_input_size = 100     # Random noise dimension coming into generator, per output vector
g_hidden_size = int(arg_dict['hs'])   # Generator complexity
g_output_size = 929    # size of generated output vector
d_input_size = 929   # Minibatch size - cardinality of distributions
d_hidden_size = g_hidden_size   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
minibatch_size = 100

MODEL = arg_dict['model'] # 'GAN', 'GAN_Gumbel_softmax', 'WGAN_GP'
var_num = [29,2,1,6,7,59,1,24,76,711,3,9,1]

# Create Generator and Discriminator object
if MODEL == 'WGAN_GP':
    G = Generator_WGANGP(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    # G.no_dropout()
else:
    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)

# Load generate model, real dataset, sampler, and decoder
state_dict = torch.load(MODEL+'/G_model.model')
G.load_state_dict(state_dict)
table = np.load('ICEM_preprocessed.npy')
var_distinct = np.load('categories.npy')

def tree(): return defaultdict(tree)
occ = tree()
for x in range(table.shape[0]):
    occ[int(table[x,7])][int(table[x,8])][int(table[x,9])]=True

mean = np.mean(table,axis=0)
std = np.std(table,axis=0)
sampler = Sampler(table, minibatch_size)
gi_sampler = sampler.get_generator_input_sampler()
Dec = Decoder(var_distinct, var_num, std,mean)


gen_input = Variable(gi_sampler(1000, g_input_size))
g_fake_data = G(gen_input)
decode = Dec.decode_record(g_fake_data)

wrong = 0
for record in decode:
    if occ[int(record[7])][int(record[8])][int(record[9])]!=True:
        wrong = wrong+1
print('wrong rate',float(wrong)/len(decode))

dr = DisclosureRisk(len(decode),table.shape[0])
dr.classification_matrix(np.array(decode),table)
print(dr.output(mode='max'))



for i in range(13):
    prelist = []
    for record in decode:
        prelist.append(record[i])
    # real data histogram 
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(121)

    a,b,c = ax.hist(table[:,i], density=True)
    # synthetic data histogram
    ax = fig.add_subplot(122)
    # print(len(prelist))
    ax.hist(prelist, bins=b, density=True)
    plt.savefig(MODEL+'/var'+str(i)+'.pdf', format='pdf')
# plt.show()