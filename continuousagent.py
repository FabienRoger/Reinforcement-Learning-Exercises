import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K

from time import time

class ContinuousAgent:
    def __init__(self, state_nb, hidden_units = 3,
                 name = 'john', discount = 0.95,
                 learning_rate_pi = 0.01, learning_rate_v = 0.1,
                 keras = True, bias = False):
        self.state_nb = state_nb #Numbers of features representing one given state
        self.learning_rate_pi = learning_rate_pi
        self.learning_rate_v = learning_rate_v
        self.discount = discount
        self.hidden_units = hidden_units #of the value function, which is a 2-layers NN
        self.bias = bias
        
        #memory of size 1
        self.states_m = np.zeros(self.state_nb)
        self.actions_m = 0
        self.rewards_m = 0
        self.next_states_m = np.zeros(self.state_nb)
        self.dones_m = False
        self.I = 1
        
        self.time = 0
        if self.bias:
            self.theta_sig = np.zeros(state_nb+1) #+bias
            self.theta_mu = np.zeros(state_nb+1)
        else:
            self.theta_sig = np.zeros(state_nb) #+bias
            self.theta_mu = np.zeros(state_nb)            
        self.v = self._create_model() 
        
        self.pi_predictor, self.pi_trainer = self._create_pi_models()
        
        self.with_keras = keras
    
    def get_action(self, state):
        s = np.array(state).reshape((-1))
        
        mu,sig = self._get_mu_sig(s)
        
        a = np.random.normal(mu,sig)
        #return min(a,20) #to avoid overflow in the exp, remember the action is ln(thrust power)
        return a
        
    def observe(self, old_state, action, reward, state, done):
        self.time += 1
        
        self.states_m = np.array(old_state).reshape((-1))
        self.actions_m = action
        self.rewards_m = reward
        self.next_states_m = np.array(state).reshape((-1))
        self.dones_m = done
    
    def update(self):
        
        #update v
        st = time()
        X = np.reshape(self.states_m, (1,-1))
        next_v = self._predict_v(self.next_states_m) if not self.dones_m else 0
        Y = np.reshape(self.rewards_m + self.discount * next_v, (1,1))
        
        
        self.v.fit(x = X, y = Y,
                    batch_size = 1, verbose=0)
        
        #update pi
        st = time()
        this_v = self._predict_v(self.states_m)
        delta = self.rewards_m + self.discount * next_v - this_v
        self.I *= self.discount
        if self.time == 1: self.I = 1 #I = gamma^t since epsiode started
        if self.dones_m: self.time = 0 #Is incremented in observe
        
        s = self.states_m 
        
        #Using the formulas found in Reinforcement Learning : An Introduction
        if self.with_keras:
            if self.bias:
                s = np.concatenate([self.states_m,[1]])
            
            with tf.GradientTape() as tape:
                #print(self.pi_trainer.output, self.pi_trainer.trainable_variables)
                ln_pi = self.pi_trainer([s.reshape((1,-1)), np.reshape(self.actions_m,(1,1))], training=True)
            ln_pi_grads = tape.gradient(ln_pi, self.pi_trainer.trainable_variables)
            
            grads = - self.I * delta * ln_pi_grads
            
            optimizer = SGD(lr=self.learning_rate_v, clipvalue=10)            
            optimizer.apply_gradients(zip(grads, self.pi_trainer.trainable_variables))            
        else:
            
            
            mu,sig = self._get_mu_sig(s) #contains the "add bias"
            
            if self.bias:
                s = np.concatenate([self.states_m,[1]])
            
            grad_ln_pi_mu = (self.actions_m - mu)/(sig**2) * s
            grad_ln_pi_sig = ((self.actions_m - mu)**2/(sig**2) - 1) * s
            
            #gradient clipping
            
            grad_ln_pi_mu = np.clip(grad_ln_pi_mu, -10,10)
            grad_ln_pi_sig = np.clip(grad_ln_pi_sig, -10,10)
            
            if self.bias:
                self.theta_mu  += np.reshape(self.learning_rate_pi * self.I * delta * grad_ln_pi_mu, (self.state_nb+1))
                self.theta_sig += np.reshape(self.learning_rate_pi * self.I * delta * grad_ln_pi_sig, (self.state_nb+1))
            else:
                self.theta_mu  += np.reshape(self.learning_rate_pi * self.I * delta * grad_ln_pi_mu, (self.state_nb))
                self.theta_sig += np.reshape(self.learning_rate_pi * self.I * delta * grad_ln_pi_sig, (self.state_nb))        
    
    def _create_model(self):
        X_in = Input((self.state_nb))
        if self.hidden_units != None:
            X = Dense(self.hidden_units, activation = 'relu')(X_in)
            X = Dense(1, activation = 'linear')(X)
        else:
            X = Dense(1, activation = 'linear')(X_in) #just a linear function
            
        model = Model(inputs = X_in, outputs = X)
        
        model.compile(optimizer=SGD(lr=self.learning_rate_v),
                          loss='mean_squared_error')
        
        return model
    
    def _create_pi_models(self):
        if self.bias:
            s = Input((self.state_nb+1))
        else:
            s = Input((self.state_nb))
        #exactly what was done manually
        mu_ln_sig = Dense(2, activation = 'linear', kernel_initializer='zeros', use_bias=False)(s)
        
        mu = Lambda(lambda tens: tens[:,0])(mu_ln_sig)
        ln_sig = Lambda(lambda tens: tens[:,1])(mu_ln_sig)
        
        ln_sig = K.clip(ln_sig, -10, 10)
        sig = K.exp(ln_sig)
        
        #s -theta-> mu and sig
        predictor = Model(inputs = s, outputs = [mu,sig])
        
        a = Input((1))
        
        p = K.exp(-0.5*(a - mu)**2 / sig**2) / ((2* np.pi*sig**2)**0.5)
        
        ln_pi = K.log(p)
        
        #s,a -theta-> log(probability density of a given s)
        trainer = Model(inputs = [s,a], outputs = ln_pi)
        
        predictor.compile(optimizer=SGD(lr=self.learning_rate_v),loss='mean_squared_error') #doesn't matter, isn't trained directly
        trainer.compile(optimizer=SGD(lr=self.learning_rate_v),loss='mean_squared_error')   #doesn't matter, isn't trained directly
        
        return predictor, trainer  
    
    def _get_mu_sig(self,s):
        if self.with_keras:
            if self.bias:
                s = np.concatenate([s,[1]])
            statea = np.array(s).reshape((1,-1))
            mu,sig = self.pi_predictor.predict(statea)
            return mu, sig
        else:
            if self.bias:
                s = np.concatenate([s,[1]]) #adds 1 for the bias
            mu = np.sum(self.theta_mu * s) #dot product
            
            dot_prod_sig = np.sum(self.theta_sig * s)
            dot_prod_sig = np.clip(dot_prod_sig, -10,10)
            sig = np.exp(np.sum(dot_prod_sig))
        
        return mu,sig
    
    def _predict_v(self, state, batch = False):
        if batch:
            return self.v.predict_on_batch(state).numpy()
        else:
            statea = np.array(state).reshape((1,-1))
            v_pred = self.v.predict(statea)
            return v_pred
    