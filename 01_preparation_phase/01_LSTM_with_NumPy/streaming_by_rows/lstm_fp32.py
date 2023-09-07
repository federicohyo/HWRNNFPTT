'''
This file from scratch implements a 3-layer network of LSTM:
    input layer - hidden layer(LSTM) - output layer

The data type used is float32, to be consistent with PyTorch
'''

import numpy as np
import copy


def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def softmax(x):  
    e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return e_x / e_x.sum(axis=1,keepdims= True) 


def log_softmax(x):  
    
    e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return np.log(e_x / e_x.sum(axis=1,keepdims= True) ) 
    

def sigmoid_derivative(values): 
    return values*(1-values)


def tanh_derivative(values): 
    return 1. - values ** 2


LSTM_DT = "float32"

def rand_arr(a, b, *args): 
    np.random.seed(1)
    return np.random.rand(*args) * (b - a) + a


class LstmParam:
    def __init__(self, layer_size, prepared_params=None, load=False, ini_range=None):
        # weight matrices

        self.ini_rng = np.sqrt(1/layer_size[1]) if ini_range is None else ini_range

        if load==False: # new parameters for training from scratch
            self.l1_wg = rand_arr(-self.ini_rng, self.ini_rng, layer_size[1], layer_size[0]+layer_size[1] ).astype(LSTM_DT)
            self.l1_wi = rand_arr(-self.ini_rng, self.ini_rng, layer_size[1], layer_size[0]+layer_size[1] ).astype(LSTM_DT) 
            self.l1_wf = rand_arr(-self.ini_rng, self.ini_rng, layer_size[1], layer_size[0]+layer_size[1] ).astype(LSTM_DT)
            self.l1_wo = rand_arr(-self.ini_rng, self.ini_rng, layer_size[1], layer_size[0]+layer_size[1] ).astype(LSTM_DT)
            self.l1_bg = rand_arr(-self.ini_rng, self.ini_rng, layer_size[1]).astype(LSTM_DT) 
            self.l1_bi = rand_arr(-self.ini_rng, self.ini_rng, layer_size[1]).astype(LSTM_DT) 
            self.l1_bf = rand_arr(-self.ini_rng, self.ini_rng, layer_size[1]).astype(LSTM_DT) 
            self.l1_bo = rand_arr(-self.ini_rng, self.ini_rng, layer_size[1]).astype(LSTM_DT) 
            self.l2_w  = rand_arr(-self.ini_rng, self.ini_rng, layer_size[2], layer_size[1] ).astype(LSTM_DT) 
            self.l2_b  = rand_arr(-self.ini_rng, self.ini_rng, layer_size[2]).astype(LSTM_DT) 

            #self.l1_wg = np.random.uniform(-self.ini_rng, selfini_rng, (layer_size[1], layr_size[0]+layer_size[1]) ).astype(LSTM_DT)
            #self.l1_wi = np.random.uniform(-self.ini_rng, self.ini_rng, (layer_size[1], layer_size[0]+layer_size[1]) ).astype(LSTM_DT) 
            #self.l1_wf = np.random.uniform(-self.ini_rng, self.ini_rng, (layer_size[1], layer_size[0]+layer_size[1]) ).astype(LSTM_DT)
            #self.l1_wo = np.random.uniform(-self.ini_rng, self.ini_rng, (layer_size[1], layer_size[0]+layer_size[1]) ).astype(LSTM_DT)
            #self.l1_bg = np.random.uniform(-self.ini_rng, self.ini_rng, layer_size[1]).astype(LSTM_DT) 
            #self.l1_bi = np.random.uniform(-self.ini_rng, self.ini_rng, layer_size[1]).astype(LSTM_DT) 
            #self.l1_bf = np.random.uniform(-self.ini_rng, self.ini_rng, layer_size[1]).astype(LSTM_DT) 
            #self.l1_bo = np.random.uniform(-self.ini_rng, self.ini_rng, layer_size[1]).astype(LSTM_DT) 
            #self.l2_w  = np.random.uniform(-self.ini_rng, self.ini_rng, (layer_size[2], layer_size[1]) ).astype(LSTM_DT) 
            #self.l2_b  = np.random.uniform(-self.ini_rng, self.ini_rng, layer_size[2]).astype(LSTM_DT) 
        else: # load trained parameters obtained somewhere else, so model can do inference right away
            self.l1_wg = prepared_params["l1_wg"] 
            self.l1_wi = prepared_params["l1_wi"] 
            self.l1_wf = prepared_params["l1_wf"] 
            self.l1_wo = prepared_params["l1_wo"] 
            self.l1_bg = prepared_params["l1_bg"] 
            self.l1_bi = prepared_params["l1_bi"] 
            self.l1_bf = prepared_params["l1_bf"] 
            self.l1_bo = prepared_params["l1_bo"] 
            self.l2_w  = prepared_params["l2_w"] 
            self.l2_b  = prepared_params["l2_b"]

            
        # initialize running mean of the weight and the bias
        self.l1_rm_wg = copy.deepcopy(self.l1_wg)
        self.l1_rm_wi = copy.deepcopy(self.l1_wi)
        self.l1_rm_wf = copy.deepcopy(self.l1_wf)
        self.l1_rm_wo = copy.deepcopy(self.l1_wo)
        self.l1_rm_bg = copy.deepcopy(self.l1_bg)
        self.l1_rm_bi = copy.deepcopy(self.l1_bi)
        self.l1_rm_bf = copy.deepcopy(self.l1_bf)
        self.l1_rm_bo = copy.deepcopy(self.l1_bo)
        self.l2_rm_w  = copy.deepcopy(self.l2_w)
        self.l2_rm_b  = copy.deepcopy(self.l2_b)

        # running estimate of the weight and the bias -- lambda
        self.l1_lbd_wg = np.zeros((layer_size[1], (layer_size[0]+layer_size[1]) )).astype(LSTM_DT) 
        self.l1_lbd_wi = np.zeros((layer_size[1], (layer_size[0]+layer_size[1]) )).astype(LSTM_DT) 
        self.l1_lbd_wf = np.zeros((layer_size[1], (layer_size[0]+layer_size[1]) )).astype(LSTM_DT) 
        self.l1_lbd_wo = np.zeros((layer_size[1], (layer_size[0]+layer_size[1]) )).astype(LSTM_DT) 
        self.l1_lbd_bg = np.zeros(layer_size[1]).astype(LSTM_DT) 
        self.l1_lbd_bi = np.zeros(layer_size[1]).astype(LSTM_DT) 
        self.l1_lbd_bf = np.zeros(layer_size[1]).astype(LSTM_DT) 
        self.l1_lbd_bo = np.zeros(layer_size[1]).astype(LSTM_DT) 
        self.l2_lbd_w  = np.zeros((layer_size[2], layer_size[1])).astype(LSTM_DT) 
        self.l2_lbd_b  = np.zeros(layer_size[2]).astype(LSTM_DT) 

        # diffs (gradient of loss function w.r.t. all parameters)
        self.l1_wg_diff = np.zeros_like(self.l1_wg).astype(LSTM_DT)
        self.l1_wi_diff = np.zeros_like(self.l1_wi).astype(LSTM_DT) 
        self.l1_wf_diff = np.zeros_like(self.l1_wf).astype(LSTM_DT) 
        self.l1_wo_diff = np.zeros_like(self.l1_wo).astype(LSTM_DT) 
        self.l1_bg_diff = np.zeros_like(self.l1_bg).astype(LSTM_DT)
        self.l1_bi_diff = np.zeros_like(self.l1_bi).astype(LSTM_DT) 
        self.l1_bf_diff = np.zeros_like(self.l1_bf).astype(LSTM_DT) 
        self.l1_bo_diff = np.zeros_like(self.l1_bo).astype(LSTM_DT) 
        self.l2_w_diff  = np.zeros_like(self.l2_w).astype(LSTM_DT)
        self.l2_b_diff  = np.zeros_like(self.l2_b).astype(LSTM_DT)



    def apply_diff(self, lr = 1):
        self.l1_wg -= (lr*  self.l1_wg_diff)
        self.l1_wi -= (lr*  self.l1_wi_diff)
        self.l1_wf -= (lr*  self.l1_wf_diff)
        self.l1_wo -= (lr*  self.l1_wo_diff)
        self.l1_bg -= (lr*  self.l1_bg_diff)
        self.l1_bi -= (lr*  self.l1_bi_diff)
        self.l1_bf -= (lr*  self.l1_bf_diff)
        self.l1_bo -= (lr*  self.l1_bo_diff)
        
        self.l2_w -= lr * self.l2_w_diff
        self.l2_b -= lr * self.l2_b_diff

        # reset diffs to zero
        self.l1_wg_diff = np.zeros_like(self.l1_wg)
        self.l1_wi_diff = np.zeros_like(self.l1_wi) 
        self.l1_wf_diff = np.zeros_like(self.l1_wf) 
        self.l1_wo_diff = np.zeros_like(self.l1_wo) 
        self.l1_bg_diff = np.zeros_like(self.l1_bg)
        self.l1_bi_diff = np.zeros_like(self.l1_bi) 
        self.l1_bf_diff = np.zeros_like(self.l1_bf) 
        self.l1_bo_diff = np.zeros_like(self.l1_bo) 

        self.l2_w_diff = np.zeros_like(self.l2_w) 
        self.l2_b_diff = np.zeros_like(self.l2_b) 


    def apply_diff_fptt(self, lr = 1, alpha = 0.001):

        # update weights
        self.l1_wg -= lr * (self.l1_wg_diff - self.l1_lbd_wg + alpha*(self.l1_wg - self.l1_rm_wg))
        self.l1_wi -= lr * (self.l1_wi_diff - self.l1_lbd_wi + alpha*(self.l1_wi - self.l1_rm_wi))
        self.l1_wf -= lr * (self.l1_wf_diff - self.l1_lbd_wf + alpha*(self.l1_wf - self.l1_rm_wf))
        self.l1_wo -= lr * (self.l1_wo_diff - self.l1_lbd_wo + alpha*(self.l1_wo - self.l1_rm_wo))
        self.l1_bg -= lr * (self.l1_bg_diff - self.l1_lbd_bg + alpha*(self.l1_bg - self.l1_rm_bg))
        self.l1_bi -= lr * (self.l1_bi_diff - self.l1_lbd_bi + alpha*(self.l1_bi - self.l1_rm_bi))
        self.l1_bf -= lr * (self.l1_bf_diff - self.l1_lbd_bf + alpha*(self.l1_bf - self.l1_rm_bf))
        self.l1_bo -= lr * (self.l1_bo_diff - self.l1_lbd_bo + alpha*(self.l1_bo - self.l1_rm_bo))

        self.l2_w -= lr * (self.l2_w_diff - self.l2_lbd_w + alpha*(self.l2_w - self.l2_rm_w))
        self.l2_b -= lr * (self.l2_b_diff - self.l2_lbd_b + alpha*(self.l2_b - self.l2_rm_b))


        # update running estimate (lambda)
        self.l1_lbd_wg -= alpha*(self.l1_wg - self.l1_rm_wg)
        self.l1_lbd_wi -= alpha*(self.l1_wi - self.l1_rm_wi)
        self.l1_lbd_wf -= alpha*(self.l1_wf - self.l1_rm_wf)
        self.l1_lbd_wo -= alpha*(self.l1_wo - self.l1_rm_wo)
        self.l1_lbd_bg -= alpha*(self.l1_bg - self.l1_rm_bg)
        self.l1_lbd_bi -= alpha*(self.l1_bi - self.l1_rm_bi)
        self.l1_lbd_bf -= alpha*(self.l1_bf - self.l1_rm_bf)
        self.l1_lbd_bo -= alpha*(self.l1_bo - self.l1_rm_bo)

        self.l2_lbd_w -= alpha*(self.l2_w - self.l2_rm_w)
        self.l2_lbd_b -= alpha*(self.l2_b - self.l2_rm_b)


        # update running mean 
        self.l1_rm_wg = 0.5*(self.l1_rm_wg + self.l1_wg) - (0.5/alpha)*self.l1_lbd_wg 
        self.l1_rm_wi = 0.5*(self.l1_rm_wi + self.l1_wi) - (0.5/alpha)*self.l1_lbd_wi
        self.l1_rm_wf = 0.5*(self.l1_rm_wf + self.l1_wf) - (0.5/alpha)*self.l1_lbd_wf
        self.l1_rm_wo = 0.5*(self.l1_rm_wo + self.l1_wo) - (0.5/alpha)*self.l1_lbd_wo
        self.l1_rm_bg = 0.5*(self.l1_rm_bg + self.l1_bg) - (0.5/alpha)*self.l1_lbd_bg
        self.l1_rm_bi = 0.5*(self.l1_rm_bi + self.l1_bi) - (0.5/alpha)*self.l1_lbd_bi
        self.l1_rm_bf = 0.5*(self.l1_rm_bf + self.l1_bf) - (0.5/alpha)*self.l1_lbd_bf
        self.l1_rm_bo = 0.5*(self.l1_rm_bo + self.l1_bo) - (0.5/alpha)*self.l1_lbd_bo

        self.l2_rm_w = 0.5*(self.l2_rm_w + self.l2_w) - (0.5/alpha)*self.l2_lbd_w
        self.l2_rm_b = 0.5*(self.l2_rm_b + self.l2_b) - (0.5/alpha)*self.l2_lbd_b


        # reset diffs to zero
        self.l1_wg_diff = np.zeros_like(self.l1_wg)
        self.l1_wi_diff = np.zeros_like(self.l1_wi) 
        self.l1_wf_diff = np.zeros_like(self.l1_wf) 
        self.l1_wo_diff = np.zeros_like(self.l1_wo) 
        self.l1_bg_diff = np.zeros_like(self.l1_bg)
        self.l1_bi_diff = np.zeros_like(self.l1_bi) 
        self.l1_bf_diff = np.zeros_like(self.l1_bf) 
        self.l1_bo_diff = np.zeros_like(self.l1_bo)

        self.l2_w_diff = np.zeros_like(self.l2_w) 
        self.l2_b_diff = np.zeros_like(self.l2_b) 


class LstmState:
    def __init__(self, n_timesteps, n_samples, layer_size, param):
        
        self.param = param
        self.xc = np.empty((n_timesteps, n_samples, (layer_size[0] + layer_size[1]) )).astype(LSTM_DT) 

        self.l1_g_input = np.empty((n_timesteps, n_samples, layer_size[1])).astype(LSTM_DT) 
        self.l1_i_input = np.empty((n_timesteps, n_samples, layer_size[1])).astype(LSTM_DT) 
        self.l1_f_input = np.empty((n_timesteps, n_samples, layer_size[1])).astype(LSTM_DT) 
        self.l1_o_input = np.empty((n_timesteps, n_samples, layer_size[1])).astype(LSTM_DT) 

        self.l1_g = np.empty((n_timesteps, n_samples, layer_size[1])).astype(LSTM_DT) 
        self.l1_i = np.empty((n_timesteps, n_samples, layer_size[1])).astype(LSTM_DT) 
        self.l1_f = np.empty((n_timesteps, n_samples, layer_size[1])).astype(LSTM_DT) 
        self.l1_o = np.empty((n_timesteps, n_samples, layer_size[1])).astype(LSTM_DT) 

        self.l1_s = np.zeros((n_timesteps+1, n_samples, layer_size[1])).astype(LSTM_DT) 
        self.l1_h = np.zeros((n_timesteps+1, n_samples, layer_size[1])).astype(LSTM_DT) 

        self.l2_h = np.empty((n_timesteps, n_samples, layer_size[2])).astype(LSTM_DT) 
        self.Y_hat = np.empty((n_timesteps, n_samples, layer_size[2])).astype(LSTM_DT) 

    def forward(self, sp, xt):

        # do forward for a piece of sequence of T steps, from the specified starting point (sp)
        # print(xt.shape)
        for t in range(sp, sp+int(xt.shape[1])):

            # for example batch size 200
            # concatenate x(t) and h(t-1)
            # (200,129) = (200,1) hstack (200,128) 
            # print(t, xt[:, t:t+1, :].shape)
            self.xc[t] = np.hstack(( np.squeeze(xt[:, t-sp:t-sp+1, :], axis=1),  self.l1_h[t-1]))

            # (200,128) = (200,129).(129, 128) + (128) 
            self.l1_g_input[t] = np.dot(self.xc[t], self.param.l1_wg.T) + self.param.l1_bg
            self.l1_i_input[t] = np.dot(self.xc[t], self.param.l1_wi.T) + self.param.l1_bi
            self.l1_f_input[t] = np.dot(self.xc[t], self.param.l1_wf.T) + self.param.l1_bf
            self.l1_o_input[t] = np.dot(self.xc[t], self.param.l1_wo.T) + self.param.l1_bo

            self.l1_g[t] = np.tanh(self.l1_g_input[t])
            self.l1_i[t] = sigmoid(self.l1_i_input[t])
            self.l1_f[t] = sigmoid(self.l1_f_input[t])
            self.l1_o[t] = sigmoid(self.l1_o_input[t])

            # (200,128) = (200,128)*(200,128) + (200,128)*(200,128)
            self.l1_s[t] = self.l1_g[t] * self.l1_i[t] + self.l1_s[t-1] * self.l1_f[t]
            # (200,128) = (200,128)*(200,128) 
            self.l1_h[t] = np.tanh(self.l1_s[t]) * self.l1_o[t]

            # (200,10) = (200,128)*(128,10) + (10)
            self.l2_h[t] = np.dot(self.l1_h[t], self.param.l2_w.T) + self.param.l2_b 
            # (200,10) = (200,10)
            self.Y_hat[t] = log_softmax(self.l2_h[t]) # followed by nll loss (negative log likelihood)
            # self.Y_hat[t] = softmax(self.l2_h[t])


class Lstm:
    def __init__(self, n_timesteps, n_samples, layer_size, param):
        self.param = param
        self.state = LstmState(n_timesteps, n_samples, layer_size, param)

        self.n_samples = n_samples
        self.layer_size = layer_size

        # derivatives: intermediate/temporary variable
        # self.dl1_h/self.dl2_h is the changing gradient of l(t) w.r.t. dh[0/1](t, t-1, t-2, ...)
        self.dl2_h = np.zeros((n_samples, self.layer_size[2])).astype(LSTM_DT)

        self.dl1_h = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT) 
        self.ds = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)
        self.di = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)
        self.df = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)
        self.dg = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)
        self.do = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)

        self.di_input = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)
        self.df_input = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)
        self.dg_input = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)
        self.do_input = np.zeros((n_samples, self.layer_size[1])).astype(LSTM_DT)



    # calculate the loss and the gradient of loss w.r.t. weight/bias
    # from given time step t
    def backward(self, t, y_label, trunc_h= None, trunc_s= None):

        # for example batch size 200 
        # (200,10) = ((200,10)-(200,10))*(200,10)
        self.dl2_h = softmax(self.state.l2_h[t]) - y_label 

        # (10, 128) = (10,200).(200,128) 
        self.param.l2_w_diff =  (np.dot(self.dl2_h.T, self.state.l1_h[t]))/self.n_samples
        # (10) = (200,10) 
        self.param.l2_b_diff = (self.dl2_h.sum(axis=0))/self.n_samples

        # (200,128) = (200,10).(10, 128)
        self.dl1_h = np.dot(self.dl2_h, self.param.l2_w)

        # earliest point of [h] (oldest) to access 
        h_ep = 0 if trunc_h is None else max(0, t-trunc_h)
        for h_step in np.arange(h_ep, t+1)[::-1]:

            # (200,128)
            # ds = self.state.l1_o[h_step] * (tanh_derivative(self.state.l1_s[h_step])) *self.dl1_h
            self.ds = self.state.l1_o[h_step] * (1 - (np.tanh(self.state.l1_s[h_step]))**2 ) *self.dl1_h
            self.do = np.tanh(self.state.l1_s[h_step]) * self.dl1_h

            self.do_input = sigmoid_derivative(self.state.l1_o[h_step]) * self.do 

            # (128,129) = (128,200).(200,129)
            self.param.l1_wo_diff += (np.dot(self.do_input.T, self.state.xc[h_step])) /self.n_samples
            # (128) = (200,128)
            self.param.l1_bo_diff += (self.do_input.sum(axis=0)) /self.n_samples
            
            
            s_ep = 0 if trunc_s is None else max(0, h_step -trunc_s)
            for s_step in np.arange(s_ep, h_step+1)[::-1]:
                # (200,128)
                self.dg = self.state.l1_i[s_step] * self.ds
                self.di = self.state.l1_g[s_step] * self.ds
                self.df = self.state.l1_s[s_step-1] * self.ds

                # (200,128)
                self.di_input = sigmoid_derivative(self.state.l1_i[s_step]) * self.di 
                self.df_input = sigmoid_derivative(self.state.l1_f[s_step]) * self.df 
                self.dg_input =    tanh_derivative(self.state.l1_g[s_step]) * self.dg

                # (128,129) = (128,200).(200,129)
                self.param.l1_wi_diff += (np.dot(self.di_input.T, self.state.xc[s_step])) /self.n_samples
                self.param.l1_wf_diff += (np.dot(self.df_input.T, self.state.xc[s_step])) /self.n_samples
                self.param.l1_wg_diff += (np.dot(self.dg_input.T, self.state.xc[s_step])) /self.n_samples

                # (128) = (200,128)
                self.param.l1_bi_diff += (self.di_input.sum(axis=0)) /self.n_samples
                self.param.l1_bf_diff += (self.df_input.sum(axis=0)) /self.n_samples       
                self.param.l1_bg_diff += (self.dg_input.sum(axis=0)) /self.n_samples       
                
                # (200,128)
                self.ds= self.ds* self.state.l1_f[s_step]
                
                if s_step == h_step: 
                    dxc = np.zeros_like(self.state.xc[s_step])

                    # (200,129) = (200,128).(128,129)
                    dxc += np.dot(self.di_input, self.param.l1_wi)
                    dxc += np.dot(self.df_input, self.param.l1_wf)
                    dxc += np.dot(self.do_input, self.param.l1_wo)
                    dxc += np.dot(self.dg_input, self.param.l1_wg)
                    # (200,128) <- (200,129) order in hstack 
                    self.dl1_h = dxc[:,self.layer_size[0]:]






 
