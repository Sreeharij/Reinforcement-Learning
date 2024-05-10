import gym
import numpy as np

env = gym.make("MountainCar-v0",render_mode = "human")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)
q_table = np.random.uniform(low = -1,high = 0,size = (DISCRETE_OS_SIZE + [env.action_space.n]))
# print(q_table)

done = False
var = 0
while not done:
    if var > 100: break
    var+=1
    action = 2
    new_state,reward,done,info,_ = env.step(action)
    print(reward,new_state)
    env.render()
env.close()