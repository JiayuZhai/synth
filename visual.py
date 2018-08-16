import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import *
import argparse
from scipy import stats
from sklearn.linear_model import LogisticRegression


def risk(table,decode,method):
    dr = DisclosureRisk(decode[0:1000,:].shape[0],table.shape[0])
    dr.classification_matrix(decode[0:1000,:],table)
    dr_min = dr.output(mode='min')
    dr_max = dr.output(mode='max')
    
    print('min:',dr_min)
    print('max:',dr_max)
    return dr_min,dr_max

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-model',default='WGAN_GP',help="The GAN model we used. Default is WGAN_GP.")
    parser.add_argument('-hs',default=50,help="Hidden unit number for both D and G.")
    parser.add_argument('-syn',default=82851,help="Synthetic records number.")
    parser.add_argument('-method',default='GAN',help='Synthesise method. Default is GAN')
    args = parser.parse_args()
    arg_dict = vars(args)

    # Load original dataset and pre-processing
    table = np.load('data/clean_data.npy')
    ppp = PreprocessPCAPostprocess(table, [True,True,False,True,True,True,True,True,False,False])
    ppp.preprocess()
    ppp.pca_fit()

    # Trained Model params 
    g_input_size = 100     # Random noise dimension coming into generator, per output vector
    g_hidden_size = int(arg_dict['hs'])   # Generator complexity
    g_output_size = int(ppp.pca.n_components_)    # size of generated output vector
    minibatch_size = 100
    syn_size = int(arg_dict['syn'])

    MODEL = arg_dict['model']
    METHOD = arg_dict['method']

    # GAN
    if METHOD == 'GAN':
        # Create Generator object
        if MODEL == 'WGAN_GP':
            G = Generator_WGANGP(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
        else:
            G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)

        # Load generate model, sampler
        state_dict = torch.load('model/'+MODEL+'/G_model.model')
        G.load_state_dict(state_dict)

        sampler = Sampler(table, minibatch_size)
        gi_sampler = sampler.get_generator_input_sampler()

        decodes = []
        for i in range(syn_size//1000+1):
            if i == 0: # synthesise every 1000 records
                sample_size = syn_size - syn_size//1000*1000
            else:
                sample_size = 1000
            if sample_size == 0:
                continue
            gen_input = Variable(gi_sampler(sample_size, g_input_size))
            g_fake_data = G(gen_input)
            decode = ppp.decode(g_fake_data.detach().numpy()) # decode with ppp object
            decodes.append(decode)


        decodes = np.concatenate(decodes,axis=0)
        # Align Synthetic dataset and original dataset
        decode = decodes[:,[0,1,7,2,3,4,5,6,8,9]]
        np.save('data/'+METHOD,decode)

    # Value Exchange
    if METHOD == 'EXCHANGE':
        table = np.load('data/clean_data.npy')
        decode = np.copy(table)
        for i in range(table.shape[1]):
            np.random.seed(i)
            np.random.shuffle(decode[:,i])
        np.save('data/'+METHOD,decode)

    # CART
    if METHOD == 'CART':
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.tree import DecisionTreeClassifier
        table = np.load('data/clean_data.npy')
        decode = np.copy(table)
        m = 2
        np.random.shuffle(decode[:,m]) # random change on 'sex'
        features = [m]
        for i in np.append(np.arange(1,10),[0]):
            if i != m: # Synthesis all other variables
                if i in [2,8,9]:
                    tree = DecisionTreeRegressor(min_samples_leaf=5)
                else:
                    tree = DecisionTreeClassifier(min_samples_leaf=5)
                tree.fit(table[:,np.array(features)],table[:,i])
                yobs = tree.predict(decode[:,np.array(features)])
                decode[:,i] = yobs
                features.append(i)
        np.save('data/'+METHOD,decode)

    if METHOD == 'compare':
        # Reload original dataset and all synthetic data
        table = np.load('data/clean_data.npy')
        decode1 = np.load('data/GAN100.npy')
        decode2 = np.load('data/EXCHANGE.npy')
        decode3 = np.load('data/CART.npy')
        decode4 = np.load('data/GAN50.npy')
        decode5 = np.load('data/GAN75.npy')

        # Create Utility object
        u = DataUtility()

        p1 = u.marginal(table,decode1,'GAN100')
        p2 = u.marginal(table,decode2,'EXCHANGE')
        p3 = u.marginal(table,decode3,'CART')
        p4 = u.marginal(table,decode4,'GAN50')
        p5 = u.marginal(table,decode5,'GAN75')

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.plot(p1,'o')
        ax.plot(p2,'+')
        ax.plot(p3,'x')
        ax.plot(p3,'go')
        ax.plot(p3,'ro')
        ax.set_xticks(np.arange(0, 10))
        variables = ['parish','sex', 'age', 'mar_stat', 'disability',
                'employ','inactive','ctry_bth','nservants','pperroom']
        ax.set_xticklabels(variables)

        plt.legend(['GAN100','EXCHANGE','CART','GAN50','GAN75'])
        plt.savefig('figure/pvalue.pdf',format='pdf')

        u1 = u.utility(table,decode1,'GAN100')
        u2 = u.utility(table,decode2,'EXCHANGE')
        u3 = u.utility(table,decode3,'CART')
        u4 = u.utility(table,decode4,'GAN50')
        u5 = u.utility(table,decode5,'GAN75')
        u6 = u.utility(table,table,'direct')

        # pMSE
        print(u1[0],u2[0],u3[0],u4[0],u5[0],u6[0])

        # CI
        CIs = np.array([u1[1],u2[1],u3[1],u4[1],u5[1]])
        CI = np.mean(CIs,axis=1)
        print(CI)

        # SD
        SDs = np.array([u1[2],u2[2],u3[2],u4[2],u5[2]])
        SD = np.mean(SDs,axis=1)
        print(SD)

        # DR
        dr = []
        dr.append(risk(table,decode1,'GAN100'))
        dr.append(risk(table,decode2,'EXCHANGE'))
        dr.append(risk(table,decode3,'CART'))
        dr.append(risk(table,decode4,'GAN50'))
        dr.append(risk(table,decode5,'GAN57'))
        dr.append(risk(table,table,'ori'))
        print('dr:',dr)

if __name__ == '__main__':
    main()