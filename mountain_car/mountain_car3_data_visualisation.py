import gym
import numpy as np
import matplotlib.pyplot as plt
import os

CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_FOLDER = os.path.dirname(CURRENT_FILE_PATH)
QTABLES_FOLDER = os.path.join(CURRENT_FOLDER,"qtables")
DELETE_CURRENT_QTABLES = True #DELETES QTABLES IF FLAG IS SET TO TRUE
SAVE_MODELS = True #SAVES MODELS IF THIS FLAG IS SET TO TRUE

if not os.path.exists(QTABLES_FOLDER):
    os.mkdir(QTABLES_FOLDER)
elif DELETE_CURRENT_QTABLES:
    for file in os.listdir(QTABLES_FOLDER):
        os.remove(os.path.join(QTABLES_FOLDER,file))

env = gym.make("MountainCar-v0",render_mode = "rgb_array")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 5000

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

bucket_size = 40
DISCRETE_OS_SIZE = [bucket_size] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low = -1,high = 0,size = (DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep' : [], 'avg' : [], 'max' : [], 'min' : []}
STATS_EVERY = 100

img_filename = f"episodes={EPISODES} window_size={bucket_size} decay_end={END_EPSILON_DECAYING} stats_every={STATS_EVERY}.png"
img_filename = os.path.join(CURRENT_FOLDER,img_filename)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))
env.close()

for episode in range(1,EPISODES+1):
    episode_reward = 0
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
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state,reward,done,info,_ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            # print(f"Succeeded on Episode {episode}")
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state
    if episode < END_EPSILON_DECAYING and episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)
    if episode % STATS_EVERY == 0:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        # print(f"Episode: {episode}, average_reward: {average_reward}, min_reward: {min(ep_rewards[-STATS_EVERY:])}, max_reward: {max(ep_rewards[-STATS_EVERY:])}, current_epsilon = {epsilon:1.2f}")
    if episode%10 == 0 and SAVE_MODELS: 
        new_file_name = os.path.join(QTABLES_FOLDER,f"{episode}-qtable.npy")
        np.save(new_file_name,q_table)
    env.close()
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = "average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = "max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = "min rewards")
plt.legend(loc=4)
plt.grid(True)
plt.show()