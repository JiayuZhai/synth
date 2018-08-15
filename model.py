import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PreprocessPCAPostprocess():
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

    # def postprocess(self, data):
        
    # def sampling(self,probs):
    #     samples = []
    #     for prob in probs:
    #         p = np.random.random_sample()
    #         i=0
    #         for j in range(len(prob)):
    #             if i+prob[j]>p:
    #                 samples.append(j)
    #                 break
    #             i = i+ prob[j]
    #     return np.array(samples)

    def decode(self, data):
        ful_dim = self.pca_inverse(data)
        length = 0
        lists = []
        for n_values in self.enc.n_values_:
            # softmax = F.softmax(torch.tensor(ful_dim[:,length:length+n_values]),dim=1)
            lists.append(np.argmax(ful_dim[:,length:length+n_values],axis=1))
            # sampler = torch.distributions.gumbel.Gumbel(0.,1.)
            # lists.append(np.argmax(ful_dim[:,length:length+n_values]+
            #     sampler.sample(sample_shape=ful_dim[:,length:length+n_values].shape),axis=1))
            length = length + n_values
        cons = self.stda.inverse_transform(ful_dim[:,[-3,-2,-1]])
        # print(cons.shape)
        for i in range(3):
            # print(cons[:,i])
            if i!=2: #make integer variables to integer
                # print(cons[:,i])
                cons[:,i] = (cons[:,i]+0.5).astype(int)
                # print(cons[:,i])
            lists.append(cons[:,i])
        return np.array(lists).T


class Sampler():
    # Sampler Class for sampling real data from table dataset and sampling noise from randn function
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
        self.var_num = [29,2,1,6,7,59,1,24,76,711,3,9,1]
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
    
    # Exponential approximate with one-hot for categorical variable 
    def exponential_approx(self, x):
        list = []
        length=0
        for i in range(len(self.var_num)):
            if self.var_num[i] == 1:
                list.append(x.narrow(1,length,1))
            else:
                # y = x.narrow(1,length,self.var_num[i])
                # shape = y.size()
                # _, ind = y.max(dim=-1)
                # y_hard = torch.zeros_like(y).view(-1, shape[-1])
                # y_hard.scatter_(1, ind.view(-1, 1), 1)
                # y_hard = y_hard.view(*shape)
                # list.append((y_hard - y).detach() + y)

                max_value, _ = x.narrow(1,length,self.var_num[i]).max(dim=1)
                neg = x.narrow(1,length,self.var_num[i])-max_value.unsqueeze(1)
                # 100.0 is the exponential factor. the larger, the better effect to approximate one-hot
                list.append(torch.pow(torch.tensor(100.0),neg)) 
                
            length = length+self.var_num[i]
        return torch.cat(list,1)
    
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        result = self.exponential_approx(self.map3(x))
        # print(result)
        return result

# ##### MODELS: Generator model and discriminator model
class Generator_basic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator_basic, self).__init__()
        # self.var_num = [29,2,1,6,7,59,1,24,76,711,3,9,1]
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        result = self.map3(x)
        return result

class Generator_Gumbel_softmax(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator_Gumbel_softmax, self).__init__()
        self.var_num = [29,2,1,6,7,59,1,24,76,711,3,9,1]
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
    
    # Exponential approximate with one-hot for categorical variable 
    def exponential_approx(self, x):
        list = []
        length=0
        for i in range(len(self.var_num)):
            if self.var_num[i] == 1:
                list.append(x.narrow(1,length,1))
            else:
                logits = x.narrow(1,length,self.var_num[i])
                # 100.0 is the exponential factor. the larger, the better to approximate one-hot
                softmax = self.gumbel_softmax_sample(logits)
                # print(softmax)
                list.append(softmax) 
            length = length+self.var_num[i]
        return torch.cat(list,1)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature=0.8):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    # def gumbel_softmax(self, logits, temperature=0.8):
    #     """
    #     input: [*, n_class]
    #     return: [*, n_class] an one-hot vector
    #     """
    #     y = self.gumbel_softmax_sample(logits, temperature)
    #     # print(y)
    #     shape = y.size()
    #     _, ind = y.max(dim=-1)
    #     y_hard = torch.zeros_like(y).view(-1, shape[-1])
    #     y_hard.scatter_(1, ind.view(-1, 1), 1)
    #     y_hard = y_hard.view(*shape)
    #     # print(y_hard)
    #     return (y_hard - y).detach() + y
    
    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        result = self.exponential_approx(self.map3(x))
        return result


class Generator_WGANGP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator_WGANGP, self).__init__()
        self.var_num = [29,2,1,6,7,59,1,24,76,711,3,9,1]
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # self.dropout1,
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            # self.dropout2,
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
        )
        self.main = main

    def no_dropout(self):
        self.dropout1.training = False
        self.dropout2.training = False

    # Sigmoid function per categorical variables
    def sigmoid_categorical(self, x):
        list = []
        length=0
        for i in range(len(self.var_num)):
            if self.var_num[i] == 1:
                list.append(x.narrow(1,length,1))
            else:
                list.append(F.sigmoid(x.narrow(1,length,self.var_num[i])))
                
            length = length+self.var_num[i]
        return torch.cat(list,1)

    def forward(self, noise):
        output = self.main(noise)
        return self.sigmoid_categorical(output)

class Generator_WGANGP_basic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator_WGANGP_basic, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # self.dropout1,
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            # self.dropout2,
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
            # nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            # nn.Dropout(p=0.2),
            nn.ReLU(True),
            # last layer should be Linear instead of sigmoid
            nn.Linear(hidden_size, output_size),
            # nn.ReLU(True),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output


class DisclosureRisk():
    def __init__(self, shape_syn, shape_ini):
        self.c = lil_matrix((shape_syn,shape_ini), dtype=int) # TODO deal with memory error, use sparse rep
        self.n = shape_ini
        self.t = shape_syn
    def output(self, mode = 'min', w=None):
        if mode == 'min':
            return self.dr_min()
        elif mode == 'max':
            return self.dr_max()
        elif mode == 'w':
            self.w = w
            # w = np.zeros(shape_syn, shape_ini) # TODO find a specific weight matrix
            return self.dr_w(w)
        else:
            print('DR_min: ', self.dr_min())
            print('DR_max: ', self.dr_max())
            # w = np.zeros(shape_syn,shape_ini) # TODO find a specific weight matrix
            return self.dr_w(w)

    def dr_min(self):
        return self.c[0,0]/self.n

    def dr_max(self):
        a = 0
        for k in range(0,self.t):
            # c = np.sum([self.c[i,k] for i in range(0,k)])
            # d = np.sum([self.c[k,j] for j in range(0,k-1)])
            c = np.sum(self.c[0:k,k])
            d = np.sum(self.c[k,0:k-1])
            a = a + 1./(k+1)*(c + d)
        b = 0
        for k in range(self.t,self.n):
            # b = b + 1./(k+1)*np.sum(self.c[i,k] for i in range(0,self.t))
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
                # print(i,j)
        # return self.c



class DataUtility():
    def __init__(self):
        pass
    def output(self):
        pass

class Decoder():
    def __init__(self, var_distinct, var_num,std,mean):
        self.var_distinct = var_distinct
        self.var_num = var_num
        self.std = std
        self.mean = mean
    # Function of decoding one-hotted record to 
    def decode_record(self,data,name=False):
        records = []
        for d in data:
            record = []
            length=0
            for j in range(13):
                if j==2 or j==6 or j==12:
                    # print(d.narrow(0,length,var_num[j]))
                    r = (d.narrow(0,length,self.var_num[j])*self.std[j]+self.mean[j]).data.storage().tolist()[0]
                    if j == 2:
                        r = int(r+0.5) #Rounding to integer age
                    record.append(r)
                else:
                    for k in self.var_distinct[j].keys():
                        index = d.narrow(0,length,self.var_num[j]).argmax()
                        if self.var_distinct[j][k]==index:
                            record.append(k if name else index)
                length = length+self.var_num[j]
            records.append(record)
        return records

# class LogisticRegression(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(LogisticRegression, self).__init__()
#         self.linear = nn.Linear(input_size, num_classes)
    
#     def forward(self, x):
#         out = self.linear(x)
#         return F.sigmoid(out)