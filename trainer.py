
def train(agent, env, epochs, max_t = 195, render = True, verbose = 1):
    hist = []
    
    for i_episode in range(epochs):
        state = env.reset()
        
        tot_reward = 0
        for t in range(max_t):
            if render:
                env.render()
                
            action = agent.get_action(state)
            
            old_state = state
            state, reward, done, info = env.step(action)
            
            if verbose >= 2:
                print(state, reward, done, info, action)
            
            agent.observe(old_state, action, reward, state, done)
            agent.update()
            
            tot_reward += reward
            
            if done:
                if verbose >= 1:
                    print("Episode {} finished after {} timesteps. G = {}".format(i_episode, t+1, tot_reward))
                hist.append((t+1, tot_reward))
                break
            elif t==max_t - 1:
                print("Episode {} finished after max_t timesteps. G = {}".format(i_episode, tot_reward))
                hist.append((t+1, tot_reward))
                break
    env.close()
    
    return hist