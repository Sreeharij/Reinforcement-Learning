import gym
import numpy as np

env = gym.make("MountainCar-v0",render_mode = "rgb_array")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
SHOW_EVERY = 5000

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low = -1,high = 0,size = (DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))
env.close()

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
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state,reward,done,info,_ = env.step(action)
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
    env.close()
