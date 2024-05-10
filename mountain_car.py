import gym
import numpy as np

env = gym.make("MountainCar-v0",render_mode = "rgb_array")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 1000000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low = -1,high = 0,size = (DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))
env.close()
steps_map = {}
for episode in range(1,EPISODES+1):
    if episode % SHOW_EVERY == 0:
        print(episode)
        env = gym.make("MountainCar-v0",render_mode = "human")
    else:
        env = gym.make("MountainCar-v0",render_mode = "rgb_array")
    initial_state = env.reset()
    discrete_state = get_discrete_state(initial_state[0])
    done = False
    steps_taken = 0
    while not done:
        steps_taken += 1
        action = np.argmax(q_table[discrete_state])
        new_state,reward,done,info,_ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            # print(f"Succeeded on Episode {episode}")
            if(steps_taken in steps_map):
                steps_map[steps_taken] = steps_map[steps_taken] + [episode]
            else:
                steps_map[steps_taken] = [episode]
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state
        
    env.close()
sorted_keys = sorted(steps_map.keys())
print(sorted_keys[0],steps_map[sorted_keys[0]])
print(sorted_keys[1],steps_map[sorted_keys[1]])
print(sorted_keys[2],steps_map[sorted_keys[2]])
print(sorted_keys[3],steps_map[sorted_keys[3]])
print(sorted_keys[4],steps_map[sorted_keys[4]])
for key in steps_map:
    if(EPISODES in steps_map[key]):
        print(key)