import random
from collections import deque

import torch
import torch.nn.functional as F
import gym
import numpy as np
import torch.optim as optim
from gym.core import Env
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_agent(
    EPSILON,
    EPSILON_DECAY, 
    LEARNING_RATE, 
    WEIGHT_DECAY, 
    BATCH_SIZE_BUFFER_SIZE, 
    POLICY_NN, 
    ITERATIONS_TO_UPDATE_TARGET_NN, 
    ACTIVATION_FUNCTION, 
    OPTIMISER_TYPE, 
    NUM_RUNS=10,
    EPISODES=250,
    DDQN=False
    ):

    torch.manual_seed(0)
    np.random.seed(0)
    
    runs_results = []
    env = gym.make('CartPole-v1')
    env.action_space.seed(0)
    
    BATCH_SIZE = BATCH_SIZE_BUFFER_SIZE[0]
    MEMORY_REPLAY_BUFFER_SIZE = BATCH_SIZE_BUFFER_SIZE[1]
    
    for _ in tqdm(range(NUM_RUNS)):
        
        np.random.seed(0)

        policy_net = DQN(POLICY_NN, ACTIVATION_FUNCTION)
        target_net = DQN(POLICY_NN, ACTIVATION_FUNCTION)
        update_target(target_net, policy_net)
        target_net.eval()
        epsilon = EPSILON

        if OPTIMISER_TYPE == 'Adam':
            optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        elif OPTIMISER_TYPE == 'RAdam':
            optimizer = optim.RAdam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        elif OPTIMISER_TYPE == 'SGD':
            optimizer = optim.SGD(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        else:
            return

        memory = ReplayBuffer(MEMORY_REPLAY_BUFFER_SIZE)

        episode_durations = []

        for i_episode in range(EPISODES):

            observation, _ = env.reset()
            state = torch.tensor(observation).float()
            done = False
            terminated = False 
            t = 0
            while not (done or terminated):

                # Select and perform an action
                action = epsilon_greedy(epsilon, policy_net, state)

                observation, reward, done, terminated, _ = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                memory.push([state, action, next_state, reward, torch.tensor([done])])

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if not len(memory.buffer) < BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    # Compute loss
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones, DDQN)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                
                if done or terminated:
                    episode_durations.append(t + 1)
                t += 1

                if epsilon < 0.01:
                    epsilon = 0.01
                else:
                    epsilon = epsilon*EPSILON_DECAY
            # Update the target network, copying all weights and biases in DQN
            if i_episode % ITERATIONS_TO_UPDATE_TARGET_NN == 0: 
                update_target(target_net, policy_net)
        
        runs_results.append(episode_durations)

    return runs_results

def get_policy_train_agent(
    EPSILON,
    EPSILON_DECAY, 
    LEARNING_RATE, 
    WEIGHT_DECAY, 
    BATCH_SIZE_BUFFER_SIZE, 
    POLICY_NN, 
    ITERATIONS_TO_UPDATE_TARGET_NN, 
    ACTIVATION_FUNCTION, 
    OPTIMISER_TYPE, 
    NUM_RUNS=1,
    EPISODES=300,
    DDQN=False
    ):

    torch.manual_seed(0)
    np.random.seed(0)
    
    runs_results = []
    env = gym.make('CartPole-v1')
    env.action_space.seed(0)
    
    BATCH_SIZE = BATCH_SIZE_BUFFER_SIZE[0]
    MEMORY_REPLAY_BUFFER_SIZE = BATCH_SIZE_BUFFER_SIZE[1]
    
    for _ in tqdm(range(NUM_RUNS)):
        
        np.random.seed(0)

        policy_net = DQN(POLICY_NN, ACTIVATION_FUNCTION)
        target_net = DQN(POLICY_NN, ACTIVATION_FUNCTION)
        update_target(target_net, policy_net)
        target_net.eval()
        epsilon = EPSILON

        if OPTIMISER_TYPE == 'Adam':
            optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        elif OPTIMISER_TYPE == 'RAdam':
            optimizer = optim.RAdam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        elif OPTIMISER_TYPE == 'SGD':
            optimizer = optim.SGD(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        else:
            return

        memory = ReplayBuffer(MEMORY_REPLAY_BUFFER_SIZE)

        episode_durations = []

        for i_episode in range(EPISODES):

            observation, _ = env.reset()
            state = torch.tensor(observation).float()
            done = False
            terminated = False 
            t = 0
            while not (done or terminated):

                # Select and perform an action
                action = epsilon_greedy(epsilon, policy_net, state)

                observation, reward, done, terminated, _ = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                memory.push([state, action, next_state, reward, torch.tensor([done])])

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if not len(memory.buffer) < BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    # Compute loss
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones, DDQN)
                    # Optimize the model
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                
                if done or terminated:
                    episode_durations.append(t + 1)
                t += 1

                if epsilon < 0.01:
                    epsilon = 0.01
                else:
                    epsilon = epsilon*EPSILON_DECAY
            # Update the target network, copying all weights and biases in DQN
            if i_episode % ITERATIONS_TO_UPDATE_TARGET_NN == 0: 
                update_target(target_net, policy_net)
        
        runs_results.append(episode_durations)

    return policy_net

def hypeparameter_exploration(baseline, parameter, parameter_values):
    
    baseline_copy = baseline.copy()
    
    all_runs_results = []
    print(f"Testing parameter: {parameter}")
    
    for p in parameter_values:
        print(f"Parameter value: {p}")
        
        baseline_copy[parameter] = p
        print(baseline_copy)
        
        runs_results = train_agent(
                        EPSILON=baseline_copy["EPSILON"],
                        EPSILON_DECAY=baseline_copy["EPSILON_DECAY"], 
                        LEARNING_RATE=baseline_copy["LEARNING_RATE"], 
                        WEIGHT_DECAY=baseline_copy["WEIGHT_DECAY"], 
                        BATCH_SIZE_BUFFER_SIZE=baseline_copy["BATCH_SIZE_BUFFER_SIZE"], 
                        POLICY_NN=baseline_copy["POLICY_NN"], 
                        ITERATIONS_TO_UPDATE_TARGET_NN=baseline_copy["ITERATIONS_TO_UPDATE_TARGET_NN"], 
                        ACTIVATION_FUNCTION=baseline_copy["ACTIVATION_FUNCTION"], 
                        OPTIMISER_TYPE=baseline_copy["OPTIMISER_TYPE"], 
                        NUM_RUNS=10
                    )
        all_runs_results.append(runs_results)
    
    return all_runs_results

def plot_hypeparameter_exploration_results(all_runs_results, parameter_values, title):

    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 22})

    for i, runs_results in enumerate(all_runs_results):

        results = torch.tensor(runs_results)
        means = results.float().mean(0)
        plt.plot(torch.arange(250), means, label=str(parameter_values[i]))
    
    plt.ylabel("Return")
    plt.xlabel("Episode")
    plt.title(title)
    plt.axhline(y = 100, color = 'r', linestyle = '--')
    plt.legend(loc='upper left')
    plt.show()

def question_2_plot(policy, speed, q, title):
    import matplotlib.patches as mpatches
    #PLOTTING POLICY
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 14})
    policy_net = policy 
    q = q    # whether q values or greedy policy is visualised

    angle_range = .2095 # you may modify this range
    omega_range = 2     # you may modify this range

    angle_samples = 100
    omega_samples = 100
    angles = torch.linspace(angle_range, -angle_range, angle_samples)
    omegas = torch.linspace(-omega_range, omega_range, omega_samples)

    greedy_q_array = torch.zeros((angle_samples, omega_samples))
    policy_array = torch.zeros((angle_samples, omega_samples))
    for i, angle in enumerate(angles):
        for j, omega in enumerate(omegas):
            state = torch.tensor([0., speed, angle, omega])
            with torch.no_grad():
                q_vals = policy_net(state)
                greedy_action = q_vals.argmax()
                greedy_q_array[i, j] = q_vals[greedy_action]
                policy_array[i, j] = greedy_action
    if q:
        plt.contourf(angles, omegas, greedy_q_array.T, cmap='cividis', levels=100)
        plt.colorbar()
    else:
        plt.contourf(angles, omegas, policy_array.T, cmap='cividis')
        yellow_patch = mpatches.Patch(color='yellow', label='1 - Push cart to right')
        blue_patch = mpatches.Patch(color='blue', label='0 - Push cart to left')
        plt.legend(handles=[yellow_patch, blue_patch]) 

    plt.xlabel("Angle")
    plt.ylabel("Angular velocity")
    plt.title(title)
    plt.show()

class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, layer_sizes:list[int], activation):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
        self.activation = activation
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor,
         DDQN)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """
      
    if DDQN:
        next_action = policy_dqn(next_states).argmax(dim=1).unsqueeze(-1)
        bellman_targets = (~dones).reshape(-1)* target_dqn(next_states).gather(1, next_action).reshape(-1) + rewards.reshape(-1)
    else:
        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)  
    
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()