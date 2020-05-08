import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

from time import time

class Agent:
    def __init__(self, state_nb, action_nb, hidden_units = 10,
                 name = 'john', epsilon = 0.2, discount = 0.95, curiosity = 0,
                 learning_rate = 0.1, memory_size = 512, update_batch_size = 64,
                 opimtimizer = 'adam',
                 random = False,
                 print_losses = False):
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.state_nb = state_nb #Numbers of features representing one given state
        self.action_nb = action_nb
        self.learning_rate = learning_rate
        self.print_losses = print_losses
        self.update_batch_size = update_batch_size
        self.discount = discount
        self.hidden_units = hidden_units
        self.curiosity = curiosity #using rewards in the same way as Dyna-Q+
        self.random = random
        self.opimtimizer = opimtimizer
        
        self.states_m = np.zeros((memory_size, self.state_nb))
        self.actions_m = np.zeros(memory_size, dtype = int)
        self.rewards_m = np.zeros(memory_size)
        self.next_states_m = np.zeros((memory_size, self.state_nb))
        self.dones_m = np.zeros(memory_size, dtype = bool)
        
        self.cursor = 0
        self.turned_once = False
        
        self.time = 0
        
        self.q = self._create_model()
    
    def get_action(self, state, mode = 'eps greedy'):
        if self.random:return np.random.randint(self.action_nb)
        
        if mode == 'eps greedy':
            probs, _ = self._actions_prob(state)
            return np.random.choice(self.action_nb, p=probs)
        
        print('action selection mode unknown')
        return None
        
    def observe(self, old_state, action, reward, state, done):
        self.time += 1
        
        self.states_m[self.cursor,:] = np.array(old_state)
        self.actions_m[self.cursor] = action
        self.rewards_m[self.cursor] = reward
        self.next_states_m[self.cursor,:] = np.array(state)
        self.dones_m[self.cursor] = done 
        
        if self.cursor+1 == self.memory_size:
            self.turned_once = True
        self.cursor = (self.cursor +1) % self.memory_size
    
    def update(self):
        if self.random:pass
        
        st = time()
        X,Y,empty = self._generate_batches()
        #print(time()-st, 'generation')
        st = time()
        if not empty:
            self.q.fit(x = X, y = Y,
                       batch_size = 32, verbose=2 if self.print_losses else 0)
        #print(time()-st, 'training')
    
    def _actions_prob(self, state, batch = False):
        if batch:
            probs = np.ones((state.shape[0], self.action_nb)) * self.epsilon/self.action_nb
            q_values = self._q_values(state, batch  = True)
            for i in range(state.shape[0]):
                probs[i,np.argmax(q_values[i,:])] = 1 - self.epsilon + self.epsilon/self.action_nb
            
             #print(probs, q_values)
        else:
            probs = [self.epsilon/self.action_nb] * self.action_nb
            q_values = self._q_values(state)
            probs[np.argmax(q_values)] = 1 - self.epsilon + self.epsilon/self.action_nb
        
        return probs, q_values
        
    
    def _create_model(self):
        X_in = Input((self.state_nb))
        if self.hidden_units != None:
            X = Dense(self.hidden_units, activation = 'relu')(X_in)
            X = Dense(self.action_nb, activation = 'linear')(X)
        else:
            X = Dense(self.action_nb, activation = 'linear')(X_in) #just a linear function
            
        model = Model(inputs = X_in, outputs = X)
        
        if self.opimtimizer == 'adam':
            model.compile(optimizer=Adam(lr=self.learning_rate),
                          loss='mean_squared_error')
        else:
            model.compile(optimizer=SGD(lr=self.learning_rate),
                          loss='mean_squared_error')
        
        return model
    
    def _q_values(self, state, batch = False):
        if batch:
            return self.q.predict_on_batch(state).numpy()
        else:
            statea = np.array(state).reshape((1,-1))
            q_values = self.q.predict(statea)
            return q_values
    
    def _generate_batches(self):
        if not self.turned_once and self.cursor < self.update_batch_size:
            return None, None, True
        
        maxidx = self.cursor if not self.turned_once else self.memory_size
        idxs = np.random.choice(maxidx, replace = False, size=self.update_batch_size)
        
        states = self.states_m[idxs,:]
        actions = self.actions_m[idxs]
        rewards = np.copy(self.rewards_m[idxs])
        next_states = self.next_states_m[idxs,:]
        dones = self.dones_m[idxs]
        
        if self.curiosity != 0:
            self._adjust_curiosity(self.memory_size//20, rewards, states)
        
        next_probs, next_q_values = self._actions_prob(next_states, batch = True)
        next_value = np.sum(next_q_values * next_probs, axis = 1)                    
        
        Y = self._q_values(states, batch = True) #by default, no error is seen
                 
        for i in range(self.update_batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.discount * next_value[i]
            
            Y[i,actions[i]] = target
          
        return states,Y, False
    
    def _adjust_curiosity(self, k, rewards, states):
        
        for i in range(self.update_batch_size):
            norms = np.sum(np.square(self.states_m - states[i]),axis = 1)
            
            k += 1 #to eliminate the point itself
            k_min_dist_sq = norms[np.argpartition(norms, k)[:k]]
            rewards[i] = rewards[i] + self.curiosity * np.sum(np.sqrt(k_min_dist_sq))  
    
        if np.random.random()<0.1:print(rewards[:10])        