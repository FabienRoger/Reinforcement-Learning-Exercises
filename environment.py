import numpy as np

class SimpleEnv:
    #Do not touch the border
    
    pos = 0
    dpos = 0.2
    
    def reset(self):
        self.pos = 0
        return [self.pos]
        
    def render(self):
        pass
    
    def step(self, action):
        direction = 1 if action == 1 else -1
        self.pos += direction * self.dpos
        done = abs(self.pos) > 1
        
        return [self.pos], 1, done, {}
    
    def close(self):
        pass

class SimpleEnv2:
    #Do touch the border ! Even if it hurts a bit
    
    pos = 0
    dpos = 1 #moves an average of 1 unit per step
    border_pos = 20 #episodes end after the border is touched, ending the suffering of the agent
    
    def reset(self):
        self.pos = 0
        return [self.pos]
        
    def render(self):
        pass
    
    def step(self, action):
        direction = 1 if action == 1 else -1
        self.pos += direction * self.dpos * np.random.random() * 2
        done = abs(self.pos) > self.border_pos
        
        return [self.pos], -abs(self.pos)/self.border_pos-5, done, {}
    
    def close(self):
        pass 

class RocketEnv:
    #A Rocket has to land on the ground, but not too fast
    #action is log_thrust power
    
    pos = 20
    speed = 0
    g = 3
    dt = 0.2
    max_pos = 500
    max_landing_speed = 3
    t = 0
    max_t = 40000000
    
    def reset(self):
        self.pos = 10
        self.speed = 0
        self.t = 0
        return [self.pos, self.speed] 
        
    def render(self):
        res = 30
        screen_pos = max(0,int(res*self.pos/self.max_pos))
        s = '|'
        for i in range(res):
            if i == screen_pos: s += '-'
            else: s += ' '
        
        print(s)
            
    def step(self, action):
        try:
            thrust = max(0,action)
        except:
            print(str(thrust) + ' out of range')
            return [self.pos, self.speed], 0, done, {}
                
        self.speed += (-self.g + thrust) * self.dt
        self.pos += self.speed
        self.t += 1
        
        done = False
        reward = 0
        if self.pos < 0:
            done = True
            if abs(self.speed) <= self.max_landing_speed:
                reward = 2 + self.max_landing_speed + self.speed
            else:
                reward = -2
        
        if self.pos > self.max_pos:
            done = True
         
        if self.t >= self.max_t:
            done = True
              
        return [self.pos, self.speed], reward, done, {}
    
    def close(self):
        pass