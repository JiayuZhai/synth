import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import *
import argparse
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import LogisticRegression
import pandas as pd
import statsmodels.api as sm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-model',default='GAN',help="The GAN model we used. Default is WGAN-GP.")
parser.add_argument('-hs',default=50,help="Hidden unit number for both D and G.")
parser.add_argument('-syn',default=82852,help="Synthetic records number.")
parser.add_argument('-visual',default=True,help='Use visdom to show visual result or not. Default is True.')
parser.add_argument('-method',default='GAN',help='Synthesise method. Default is GAN')
args = parser.parse_args()
arg_dict = vars(args)

visual = bool(arg_dict['visual'])
if visual:
    import visdom
    vis = visdom.Visdom()
# Load original dataset and pre-processing
table = np.load('ICEM_preprocessed_new.npy')
ppp = PreprocessPCAPostprocess(table, [True,True,False,True,True,True,True,True,False,False])
ppp.preprocess()
ppp.pca_fit()

# Trained Model params 
g_input_size = 100     # Random noise dimension coming into generator, per output vector
g_hidden_size = int(arg_dict['hs'])   # Generator complexity
# print(ppp.pca.n_components_)
g_output_size = int(ppp.pca.n_components_)    # size of generated output vector
minibatch_size = 100
syn_size = int(arg_dict['syn'])

MODEL = arg_dict['model'] # 'GAN', 'GAN_Gumbel_softmax', 'WGAN_GP'
METHOD = arg_dict['method']
var_num = [29,2,1,6,7,59,1,24,76,711,3,9,1]
variables = ['parish','sex', 'age', 'mar_stat', 'disability','employ','inactive','ctry_bth','nservants','pperroom']

# GAN
if METHOD == 'GAN':
    # Create Generator object
    if MODEL == 'WGAN_GP':
        G = Generator_WGANGP_basic(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
    else:
        G = Generator_basic(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)

    # Load generate model, sampler
    state_dict = torch.load(MODEL+'/G_model.model')
    G.load_state_dict(state_dict)

    var_distinct = np.load('categories.npy')

    sampler = Sampler(table, minibatch_size)
    gi_sampler = sampler.get_generator_input_sampler()

    decodes = []
    for i in range(syn_size//1000+1):
        if i == 0:
            sample_size = syn_size - syn_size//1000*1000
        else:
            sample_size = 1000
        if sample_size == 0:
            continue
        gen_input = Variable(gi_sampler(sample_size, g_input_size))
        g_fake_data = G(gen_input)
        decode = ppp.decode(g_fake_data.detach().numpy()) #Decode with ppp object
        decodes.append(decode)


    decodes = np.concatenate(decodes,axis=0)
    # Align Synthetic dataset and original dataset
    decode = decodes[:,[0,1,7,2,3,4,5,6,8,9]]
    np.save(METHOD,decode)

# Value Exchange
if METHOD == 'EXCHANGE':
    table = np.load('ICEM_preprocessed_new.npy')
    decode = np.copy(table)
    for i in range(table.shape[1]):
        np.random.seed(i)
        np.random.shuffle(decode[:,i])
    np.save(METHOD,decode)

# CART
if METHOD == 'CART':
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeClassifier
    table = np.load('ICEM_preprocessed_new.npy')
    decode = np.copy(table)
    m = 2
    np.random.shuffle(decode[:,m])
    features = [m]
    for i in np.append(np.arange(1,10),[0]):
        if i != m:
            if i in [2,8,9]:
                tree = DecisionTreeRegressor(min_samples_leaf=5)
            else:
                tree = DecisionTreeClassifier(min_samples_leaf=5)
            tree.fit(table[:,np.array(features)],table[:,i])
            yobs = tree.predict(decode[:,np.array(features)])
            decode[:,i] = yobs
            # print('y:',decode[0:5])
            features.append(i)
            # break
    np.save(METHOD,decode)

def marginal(table,decode,method):
    
    # Histogram for 10 variables
    fig = plt.figure(figsize=(8,8))
    p_list = []
    for i in range(10):
        ax = fig.add_subplot(5,2,i+1)
        
        if i in [2,8,9]:
            bins = None
            # stat,p = stats.ks_2samp(decode[:,i], table[np.random.randint(table.shape[0],size=decode.shape[0]),i])
            # stat,p = stats.ks_2samp(decode[np.random.randint(table.shape[0],size=1000),i], table[np.random.randint(table.shape[0],size=1000),i])
            # print(decode[0:5,i],table[0:5,i])
            # p_list.append(p)
            # k = k+1
            # n1 = decode[:,i].shape[0]
            # n2 = table[:,i].shape[0]
            # n1 = len(decode[:,i])
            # n2 = len(table[:,i])
            # data1 = np.sort(decode[:,i])
            # data2 = np.sort(table[:,i])
            # data_all = np.concatenate([data1,data2])
            # cdf1 = np.searchsorted(data1,data_all,side='right')/(1.0*n1)
            # cdf2 = (np.searchsorted(data2,data_all,side='right'))/(1.0*n2)
            # ax2 = fig2.add_subplot(3,1,k)
            # ax2.plot(cdf1)
            # ax2.plot(cdf2)
            # print('ks: s={0};p={1:.2f}'.format(stat,p))
            # print(p,',')
        else:
            # bins = max(int(np.max(table[:,i]))+1,int(np.max(decode[:,i]))+1)
            bins = int(np.max(table[:,i]))+1
            
        # a,b,c = ax.hist(np.concatenate([np.expand_dims(table[:,i],axis=1),np.expand_dims(decode[:,i],axis=1)],axis=1),bins=bins)
        a,b,c = ax.hist([table[:,i],decode[:,i]],bins=bins,density=False,color = ['#000000','#808080'])
        # 
            # print(a[1],a[0])
            # pass
        contingency = np.array(a)#*500/table.shape[0]
        print(contingency)
            # exp = stats.contingency.expected_freq(contingency)
            # print(exp)
            # print('contingency:',a)
        stat,p,dof,__, = stats.chi2_contingency(contingency,correction=True)
            # stat,p = stats.chisquare(contingency.flatten(),f_exp=exp.flatten(),ddof=[0,contingency.shape[1]-1])
        p_list.append(p)
        print('chi2: s={0:.2f};\tp={1:.2f};dof={2}'.format(stat,p,dof))
            # print('chi2: s={0};p={1:.2f}'.format(stat,p))
            # print('chi2:',p)
            # print(p,',')
        # if i not in [2,8,9]:    
        #     s = 'Dataset & '
        #     for bi in b[1:]:
        #         s = s + '{} & '.format(int(bi))
        #     s = s[:-2] + '\\\\\\hline\n\\hline\no & '
        #     for j in range(len(a[0])):
        #         s = s + '{} & '.format(int(a[0][j]))
        #     s = s[:-2] + '\\\\\\hline\ns & '
        #     for j in range(len(a[1])):
        #         s = s + '{} & '.format(int(a[1][j]))
        #     s = s[:-2] + '\\\\\\hline'
        #     print(s)

        ax.set_title(variables[i])
        ax.set_ylim(0,sum(a[1]))
        ax.set_yticks(np.arange(0, sum(a[1]), sum(a[1])*0.3))
        ax.set_yticklabels(['0%','30%','60%','90%'])
        ax.legend(['Real','Synthetic'])
        # print(i,a,b,bins)

    plt.tight_layout()
    ax.legend(['Real','Synthetic'])
    if visual:
        vis.matplot(plt)
    plt.savefig(method+'vars.pdf', format='pdf')
    # plt.show()
    return p_list

def logisticReg(predictors, label):
    logit = LogisticRegression(C=1)#
    # Fit model. Let X_train = matrix of predictors, y_train = matrix of variable.
    # NOTE: Do not include a column for the intercept when fitting the model.
    resLogit = logit.fit(predictors, label)
    # print(resLogit.coef_,resLogit.intercept_,resLogit.n_iter_)

    # Calculate matrix of predicted class probabilities. 
    # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
    # print(resLogit.classes_)
    acc = resLogit.score(predictors, label)
    predProbs = np.matrix(resLogit.predict_proba(predictors))
    # print('sklearn',predProbs[:5,:])
    # Design matrix -- add column of 1's at the beginning of your X_train matrix
    X_design = np.hstack((np.ones(shape = (predictors.shape[0],1)), predictors))

    # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
    from scipy.sparse import dia_matrix
    # dia_matrix((data, offsets), shape=(4, 4)).toarray()
    V = dia_matrix((np.multiply(predProbs[:,0], predProbs[:,1]).A1,np.array([0])), shape = (X_design.shape[0], X_design.shape[0]))
    # np.fill_diagonal(V, )
    
    # Covariance matrix
    covLogit = np.linalg.pinv(X_design.T * V * X_design)

    # Standard errors
    # print(np.diag(covLogit))
    std = np.sqrt(np.diag(covLogit))
    # print ("Standard errors: ", np.sqrt(np.diag(covLogit)))

    logitParams = np.insert(resLogit.coef_, 0, resLogit.intercept_)

    bse = std
    q = stats.norm.ppf(1-0.05/2)

    lower = logitParams - q * bse
    upper = logitParams + q * bse
    return np.array([logitParams,std,lower,upper]), acc

def utility(table,decode,method):
    # Data Utility
    lr = LogisticRegression(C=1)

    ppp = PreprocessPCAPostprocess(table, [True,True,False,True,True,True,True,True,False,False])
    ppp.preprocess()
    ori = ppp.table
    syn = ppp.preprocess_data(decode)

    tol = np.concatenate([ori,syn])
    label = np.concatenate([np.ones(ori.shape[0],dtype=int),np.zeros(ori.shape[0],dtype=int)])
    lr.fit(tol,label)
    # score = np.amax(lr.predict_proba(tol), axis=1)
    score2 = lr.predict_proba(tol)
    score = score2[:,0]
    print(score2.shape)
    pMSE = np.sum((score-0.5)**2)/tol.shape[0]
    print('pMSE:', pMSE)

    SU_ori = ori#table[np.random.randint(table.shape[0],size=decode.shape[0]),:]
    SU_syn = syn#decode
    # print('0',SU_syn[0,0:29].shape)
    # print('1',SU_syn[0,29:31].shape)
    # print('2',SU_syn[0,31:37].shape)
    # print('3',SU_syn[0,37:44].shape)
    # print('4',SU_syn[0,44:47].shape)
    # print('5',SU_syn[0,47:56].shape)
    # print('6',SU_syn[0,56:115].shape)
    # print('7',SU_syn[0,115].shape)
    # print('8',SU_syn[0,116].shape)
    # print('9',SU_syn[0,117].shape)

    pred = np.append(np.arange(31,35),[np.arange(44,47),[115,116,117]])
    syn_lr_res, acc = logisticReg(SU_syn[:,pred],SU_syn[:,32])
    print('syn_acc: ',acc)
    ori_lr_res, acc = logisticReg(SU_ori[:,pred],SU_ori[:,32])
    print('ori_acc: ',acc)

    CI_ori = ori_lr_res.T[:,[2,3]]
    CI_syn = syn_lr_res.T[:,[2,3]]
    beta_ori = ori_lr_res.T[:,0]
    beta_syn = syn_lr_res.T[:,0]
    SE_ori = ori_lr_res.T[:,1]
    SE_syn = syn_lr_res.T[:,1]

    variables_CI = ['Bias term' ,'mar_stat-Single' ,'mar_stat-Married' ,'mar_stat-Widowed','mar_stat-Married spouse absent',
    'employ-W','employ-blank','employ-E','age','nservants','pperroom']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(CI_ori)):
        ax.plot(CI_ori[i],[i+1,i+1],color ='#000000')
        ax.plot(beta_ori[i],i+1,'*',color ='#000000' )
        ax.plot(CI_syn[i],[i+1.2,i+1.2],color ='#808080')
        ax.plot(beta_syn[i],i+1.2,'o',color ='#808080')
    T=np.arange(1,len(CI_ori)+1)
    ax.set_yticks(T)
    labels=[variables_CI[i] for i in range(len(CI_ori))]
    ax.set_yticklabels(labels)
    ax.legend(['original CI','original weight','synthetic CI','synthetic weight'],loc='best')
    plt.tight_layout()
    if visual:
        vis.matplot(plt)
    plt.savefig(method+'CI.pdf',format='pdf')

    IO = []
    for i in range(len(CI_ori)):
        a1 = min(CI_ori[i,1],CI_syn[i,1]) - max(CI_ori[i,0],CI_syn[i,0]) ;print('a1',a1)
        b1 = CI_ori[i,1]-CI_ori[i,0] ;print('b1',b1)
        b2 = CI_syn[i,1]-CI_syn[i,0] ;print('b2',b2)
        if np.isnan(a1) or np.isinf(a1):
            IO.append(0)
        elif np.isnan(b2) or b2==0.:
            IO.append(0.5 * (a1/b1))
        elif np.isnan(b1) or b1==0.:
            IO.append(0.5 * (a1/b2))
        else:
            IO.append(0.5 * (a1/b1+a1/b2))
    print('IO', IO)

    SD = []
    for i in range(len(beta_ori)):
        se_beta_ori = SE_ori[i]
        SD.append(abs(beta_ori[i] - beta_syn[i])/se_beta_ori)
    print('SD', SD)
    return pMSE, IO, SD

def risk(table,decode,method):
    dr = DisclosureRisk(decode[0:1000,:].shape[0],table.shape[0])
    # print()
    dr.classification_matrix(decode[0:1000,:],table)
    dr_min = dr.output(mode='min')
    dr_max = dr.output(mode='max')
    
    print('min:',dr_min)
    print('max:',dr_max)
    return dr_min,dr_max

if METHOD == 'compare':
    # Reload original dataset
    table = np.load('ICEM_preprocessed_new.npy')
    decode1 = np.load('GAN100.npy')
    decode2 = np.load('EXCHANGE.npy')
    decode3 = np.load('CART.npy')
    decode4 = np.load('GAN50.npy')
    decode5 = np.load('GAN75.npy')

    p1 = marginal(table,decode1,'GAN100')
    # p2 = marginal(table,decode2,'EXCHANGE')
    # p3 = marginal(table,decode3,'CART')
    # p4 = marginal(table,decode4,'GAN50')
    # p5 = marginal(table,decode5,'GAN75')

    # fig = plt.figure(figsize=(10,6))
    # ax = fig.add_subplot(111)
    # ax.plot(p1,'o')
    # ax.plot(p2,'+')
    # ax.plot(p3,'x')
    # ax.plot(p3,'go')
    # ax.plot(p3,'ro')
    # ax.set_xticks(np.arange(0, 10))
    # ax.set_xticklabels(variables)

    # plt.legend(['GAN100','EXCHANGE','CART','GAN50','GAN75'])
    # plt.savefig('pvalue.pdf',format='pdf')

    # u1 = utility(table,decode1,'GAN100')
    # u2 = utility(table,decode2,'EXCHANGE')
    # u3 = utility(table,decode3,'CART')
    # u4 = utility(table,decode4,'GAN50')
    # u5 = utility(table,decode5,'GAN75')
    # u6 = utility(table,table,'direct')

    # # pMSE
    # print(u1[0],u2[0],u3[0],u4[0],u5[0],u6[0])

    # CI
    # CIs = np.array([u1[1],u2[1],u3[1],u4[1],u5[1]])
    # CI = np.mean(CIs,axis=1)
    # print(CI)
    # fig = plt.figure()
    # plt.plot(u1[1],'o')
    # plt.plot(u2[1],'o')
    # plt.plot(u3[1],'o')
    # plt.plot(u4[1],'o')
    # plt.plot(u5[1],'o')

    # plt.legend(['GAN100','EXCHANGE','CART','GAN50','GAN75'])
    # plt.savefig('CIcom.pdf',format='pdf')

    #SD
    # SDs = np.array([u1[2],u2[2],u3[2],u4[2],u5[2]])
    # SD = np.mean(SDs,axis=1)
    # print(SD)
    # fig = plt.figure()
    # plt.plot(u1[2],'o')
    # plt.plot(u2[2],'o')
    # plt.plot(u3[2],'o')
    # plt.plot(u4[2],'o')
    # plt.plot(u5[2],'o')

    # plt.legend(['GAN100','EXCHANGE','CART','GAN50','GAN75'])
    # plt.savefig('SDcom.pdf',format='pdf')

    ## DR
    # dr = []
    # dr.append(risk(table,decode1,'GAN100'))
    # dr.append(risk(table,decode2,'EXCHANGE'))
    # dr.append(risk(table,decode3,'CART'))
    # dr.append(risk(table,decode4,'GAN50'))
    # dr.append(risk(table,decode5,'GAN57'))
    # dr = risk(table,table,'ori')
    # print('dr:',dr)

