import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class PreprocessPCAPostprocess():
    """Preprocess and PCA and postprocess class"""
    def __init__(self, table, features_attr):
        self.table = table
        self.features_attr = features_attr
        self.con_features = [i for i in range(len(self.features_attr)) if not self.features_attr[i]]
        print(self.con_features)
        self.con_features = [2,8,9]

        self.stda = StandardScaler()
        self.stda.fit(self.table[:,self.con_features])
        print(self.table.shape,self.features_attr)

        self.enc = OneHotEncoder(categorical_features=self.features_attr)
        self.enc.fit(self.table)
        print('max number', self.enc.n_values_)
        self.pca = None

    def preprocess(self):
        # standardisation
        self.table[:,self.con_features] = self.stda.transform(self.table[:,self.con_features])
        # one hot encoding
        self.table = self.enc.transform(self.table).todense()

    def preprocess_data(self,data):
        # standardisation
        data[:,self.con_features] = self.stda.transform(data[:,self.con_features])
        # one hot encoding
        data = self.enc.transform(data).todense()
        return data

    def pca_fit(self, n_components=None):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.table)
        print(self.pca.components_.shape)

    def pca_transform(self):
        if self.pca == None:
            raise 'no fitted pca model, use pca_fit() first'
        return self.pca.transform(self.table)

    def pca_inverse(self, data):
        if self.pca == None:
            raise 'no fitted pca model, use pca_fit() first'
        return self.pca.inverse_transform(data)

    def decode(self, data):
        ful_dim = self.pca_inverse(data)
        length = 0
        lists = []
        for n_values in self.enc.n_values_:
            lists.append(np.argmax(ful_dim[:,length:length+n_values],axis=1))
            length = length + n_values
        cons = self.stda.inverse_transform(ful_dim[:,[-3,-2,-1]])
        for i in range(3):
            if i!=2: #make integer variables to integer
                cons[:,i] = (cons[:,i]+0.5).astype(int)
            lists.append(cons[:,i])
        return np.array(lists).T


class Sampler():
    ''' Sampler Class for sampling real data from table dataset and sampling noise from randn function ''' 
    def __init__(self, table, minibatch_size):
        self.sampler_counter = 0
        self.table = table
        self.minibatch_size = minibatch_size
        
    def get_distribution_sampler(self):
        if self.sampler_counter+self.minibatch_size > self.table.shape[0]:
            np.random.shuffle(self.table)
            self.sampler_counter = 0
        self.sampler_counter = self.sampler_counter + self.minibatch_size
        return lambda n: torch.Tensor(self.table[self.sampler_counter-self.minibatch_size:self.sampler_counter,:])

    def get_generator_input_sampler(self):
        return lambda m, n: torch.randn(m, n)

# ##### MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        result = self.map3(x)
        return result

class Generator_WGANGP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator_WGANGP, self).__init__()
        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
        )
        self.main = main

    def no_dropout(self):
        self.dropout1.training = False
        self.dropout2.training = False

    def forward(self, data):
        return self.main(data)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
                
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))

class Discriminator_WGANGP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator_WGANGP, self).__init__()
        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            # last layer should be Linear instead of sigmoid
            nn.Linear(hidden_size, output_size),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output


# Evaluation measures
class DisclosureRisk():
    """Disclosure risk measure class"""
    def __init__(self, shape_syn, shape_ini):
        self.c = lil_matrix((shape_syn,shape_ini), dtype=int)
        self.n = shape_ini
        self.t = shape_syn

    def output(self, mode = 'min', w=None):
        if mode == 'min':
            return self.dr_min()
        elif mode == 'max':
            return self.dr_max()
        elif mode == 'w':
            self.w = w
            return self.dr_w(w)
        else:
            print('DR_min: ', self.dr_min())
            print('DR_max: ', self.dr_max())
            return self.dr_w(w)

    def dr_min(self):
        return self.c[0,0]/self.n

    def dr_max(self):
        a = 0
        for k in range(0,self.t):
            c = np.sum(self.c[0:k,k])
            d = np.sum(self.c[k,0:k-1])
            a = a + 1./(k+1)*(c + d)
        b = 0
        for k in range(self.t,self.n):
            b = b + 1./(k+1)*np.sum(self.c[0:self.t,k])
        return (a+b)/self.n

    def dr_w(self, w):
        a = 0
        for k in range(0,self.t):
            c = np.sum(w[i,k] * self.c[i,k] for i in range(0,k))
            d = np.sum(w[k,j] * self.c[k,j] for j in range(0,k-1))
            a = a + 1./(k+1)*(c + d)
        b = 0
        for k in range(self.t,self.n):
            b = b + 1./(k+1)*np.sum(w[i,k] * self.c[i,k] for i in range(1,self.t+1))
        return (a+b)/self.n

    def classification_matrix(self, synd, inid, key_attrs = [0,1,2,3,5]):
        for record in synd:
            i = -1
            j = -1
            for synd_r in synd:
                if (synd_r[key_attrs] == record[key_attrs]).all():
                    i = i+1
            for inid_r in inid:
                if (inid_r[key_attrs] == record[key_attrs]).all():
                    j = j+1
            if i>=0 and j >=0:
                self.c[i,j] = self.c[i,j] + 1


class DataUtility():
    ''' Utility measures include Marginal distribution, General Utility and Specific Utility'''
    def __init__(self):
        self.variables = ['parish','sex', 'age', 'mar_stat', 'disability',
                'employ','inactive','ctry_bth','nservants','pperroom']

    def output(self):
        pass
    def marginal(self,table,decode,method):
        # Histogram for 10 variables
        fig = plt.figure(figsize=(8,8))
        p_list = []
        for i in range(10):
            ax = fig.add_subplot(5,2,i+1)
            
            if i in [2,8,9]:
                bins = None
            else:
                bins = int(np.max(table[:,i]))+1    
            a,b,c = ax.hist([table[:,i],decode[:,i]],bins=bins,density=False,color = ['#000000','#808080'])
            contingency = np.array(a)*500/table.shape[0]
            stat,p,dof,__, = stats.chi2_contingency(contingency,correction=True)
            p_list.append(p)
            print('chi2: s={0:.2f};\tp={1:.2f};dof={2}'.format(stat,p,dof))

            ax.set_title(self.variables[i])
            ax.set_ylim(0,sum(a[1]))
            ax.set_yticks(np.arange(0, sum(a[1]), sum(a[1])*0.3))
            ax.set_yticklabels(['0%','30%','60%','90%'])
            ax.legend(['Real','Synthetic'])
        plt.tight_layout()
        ax.legend(['Real','Synthetic'])
        plt.savefig('figure/'+method+'vars.pdf', format='pdf')
        return p_list

    def __logisticReg(self,predictors, label):
        logit = LogisticRegression(C=1)
        # fit a LR model
        resLogit = logit.fit(predictors, label)

        # Construct covariance matrix
        acc = resLogit.score(predictors, label)
        predProbs = np.matrix(resLogit.predict_proba(predictors))
        X_design = np.hstack((np.ones(shape = (predictors.shape[0],1)), predictors))
        from scipy.sparse import dia_matrix
        V = dia_matrix((np.multiply(predProbs[:,0], predProbs[:,1]).A1,np.array([0])), shape = (X_design.shape[0], X_design.shape[0]))
        covLogit = np.linalg.pinv(X_design.T * V * X_design)

        # Standard errors
        std = np.sqrt(np.diag(covLogit))

        logitParams = np.insert(resLogit.coef_, 0, resLogit.intercept_)

        # 0.05 p-value
        q = stats.norm.ppf(1-0.05/2)
        # CI
        lower = logitParams - q * std
        upper = logitParams + q * std
        return np.array([logitParams,std,lower,upper]), acc

    def utility(self,table,decode,method):
        # General Utility - pMSE
        lr = LogisticRegression(C=1)

        ppp = PreprocessPCAPostprocess(table, [True,True,False,True,True,True,True,True,False,False])
        ppp.preprocess()
        ori = ppp.table
        syn = ppp.preprocess_data(decode)

        tol = np.concatenate([ori,syn])
        label = np.concatenate([np.ones(ori.shape[0],dtype=int),np.zeros(ori.shape[0],dtype=int)])
        # Fit a LR model by 
        lr.fit(tol,label)
        scores = lr.predict_proba(tol)
        score = scores[:,0]
        # print(score2.shape)
        pMSE = np.sum((score-0.5)**2)/tol.shape[0]
        print('pMSE:', pMSE)

        # Specific Utility - Do regression
        pred = np.append(np.arange(31,35),[np.arange(44,47),[115,116,117]])
        syn_lr_res, acc = self.__logisticReg(syn[:,pred],syn[:,32])
        print('syn_acc: ',acc)
        ori_lr_res, acc = self.__logisticReg(ori[:,pred],ori[:,32])
        print('ori_acc: ',acc)

        # Specific Utility - CI overlap and SD
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
        plt.savefig('figure/'+method+'CI.pdf',format='pdf')

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

