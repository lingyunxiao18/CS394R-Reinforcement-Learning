import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Env:
    def __init__(self):
        self.grid_size = 10
        self.num_evacuating_facilities = 2
        self.num_receiving_facilities = 5
        self.num_actions = 4
        self.num_evacuee_vehicle = 0
        self.vehicle_position = [3, 2]
        self.evacuating_facilities = [[7, 2], [9, 1]]
        self.receiving_facilities = [[5, 5], [9, 3], [7, 6], [1, 9], [2, 2]]
        self.num_evacuee_evacuating = [36, 29]
        self.remaining_receiving_capacity = [16, 37, 22, 44, 43] # Fixed
        self.num_evacuee_receiving = [0, 0, 0, 0, 0]
        self.time_step = 0
        # self.max_time_steps = 1000

    def reset(self):
        self.vehicle_position = [3, 2]
        self.evacuating_facilities = [[7, 2], [9, 1]]
        self.receiving_facilities = [[5, 5], [9, 3], [7, 6], [1, 9], [2, 2]]
        self.num_evacuee_evacuating = [36, 29]
        self.num_evacuee_receiving = [0, 0, 0, 0, 0]
        self.time_step = 0
        return self._get_state()

    def step(self, action):
        self.time_step += 1
        reward = -1
        done = False

        # Update vehicle position
        if action == 0:  # Up
            self.vehicle_position[0] = max(0, self.vehicle_position[0] - 1)
        elif action == 1:  # Down
            self.vehicle_position[0] = min(self.grid_size - 1, self.vehicle_position[0] + 1)
        elif action == 2:  # Left
            self.vehicle_position[1] = max(0, self.vehicle_position[1] - 1)
        elif action == 3:  # Right
            self.vehicle_position[1] = min(self.grid_size - 1, self.vehicle_position[1] + 1)
            
        # Check if the vehicle is at an evacuating facility
        for i, facility in enumerate(self.evacuating_facilities):
            if self.vehicle_position == facility:
                # Empty vehicle
                if self.num_evacuee_vehicle == 0:
                    if self.num_evacuee_evacuating[i] > 0:
                        # Pick up an evacuee
                        self.num_evacuee_evacuating[i] -= 1
                        self.num_evacuee_vehicle += 1
                        # print(f"Picked up an evacuee at {facility}!")
                        reward += 10*(self.num_evacuee_evacuating[i])
                    # else:
                    #     # print(f"Empty vehicle arrives at empty evacuating facility {facility}.")
                    #     reward -= 1
                # else:
                #     # Add negative signal if an empty vehicle arrives at the evacuating facility
                #     # print(f"Loaded vehicle arrives at evacuating facility {facility}.")
                #     reward -= 1
                break

        # Check if the vehicle is at a receiving facility
        for i, facility in enumerate(self.receiving_facilities):
            if self.vehicle_position == facility:
                # Loaded vehicle
                if self.num_evacuee_vehicle > 0: 
                    if self.num_evacuee_receiving[i] < self.remaining_receiving_capacity[i]:
                        # Drop off an evacuee
                        self.num_evacuee_receiving[i] += 1
                        self.num_evacuee_vehicle -= 1
                        # print(f"Dropped off an evacuee at {facility}!")
                        reward += 5*(self.remaining_receiving_capacity[i] - self.num_evacuee_receiving[i])
                    # else:
                    #     # Add negative signal if a loaded vehicle arrives at the full receiving facility
                    #     # print(f"Loaded vehicle arrives at full receiving facility {facility}.")
                    #     reward -= 1
                # else:
                #     # Add negative signal if an empty vehicle arrives at any receiving facility
                #     # print(f"Empty vehicle arrives at receiving facility {facility}.")
                #     reward -= 1
                break

        # Check if all evacuees have been transferred and the vehicle has returned to the depot
        if (sum(self.num_evacuee_evacuating) + self.num_evacuee_vehicle) == 0:
            reward += 1000
            # print("Done with the mission!")
            done = True

        # Check if any evacuee is left behind but the vehicle has returned to the depot
        # if sum(self.remaining_evacuees) != 0 and self.vehicle_position == [0, 0]:
        #     reward -= 10
        #     print("Failed.")
        #     done = True

        # Check if max time steps reached
        # if self.time_step >= self.max_time_steps:
        #     # print("Max time steps reached!")
        #     done = True

        return self._get_state(), reward, done

    # The state consists of three parts: the vehicle's location, the number of remaining evacuees in each
    # evacuating facility, and the number of remaining capacity in each receiving facility
    def _get_state(self):
        vehicle_state = [0] * self.grid_size * self.grid_size 
        vehicle_state[self.vehicle_position[0] * self.grid_size + self.vehicle_position[1]] = 1
        state = np.array(vehicle_state + self.num_evacuee_evacuating + self.num_evacuee_receiving)
        return state

    # def render(self):
    #     grid = np.zeros((self.grid_size, self.grid_size))
    #     grid[self.vehicle_position[0], self.vehicle_position[1]] = 1
    #     for facility in self.evacuating_facilities:
    #         grid[facility[0], facility[1]] = 0.5
    #     for facility in self.receiving_facilities:
    #         grid[facility[0], facility[1]] = 0.8
    #     print(grid)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.rl = nn.LeakyReLU()  
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, action_mask=None):
        x = self.rl(self.fc1(x))  
        x = self.dropout(x)
        x = self.rl(self.fc2(x))  
        logits = self.fc_actor(x)
        # if action_mask is not None:
        #     logits += (1 - action_mask) * -1e9
        value = self.fc_critic(x)
        return logits, value
       

class Actor_Critic_Agent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=1, max_grad_norm=0.5):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits, values = self.actor_critic(state)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)
        log_probs = F.log_softmax(logits, dim=-1)
        return action.item(), log_probs, values

    # def mask(self, action_dim, position):
    #     action_mask = torch.ones(action_dim)
    #     if position[0] == 0:
    #         action_mask[0] = 0  # Disable 'Up' action if vehicle is at the top row
    #     if position[0] == env.grid_size - 1:
    #         action_mask[1] = 0  # Disable 'Down' action if vehicle is at the bottom row
    #     if position[1] == 0:
    #         action_mask[2] = 0  # Disable 'Left' action if vehicle is at the leftmost column
    #     if position[1] == env.grid_size - 1:
    #         action_mask[3] = 0  # Disable 'Right' action if vehicle is at the rightmost column
    #     return action_mask

    def update(self, log_probs, values, rewards):
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        actor_loss = (-log_probs * (returns - values.detach())).mean()
        critic_loss = F.smooth_l1_loss(values, returns.unsqueeze(1))
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

if __name__ == "__main__":
    # Initialize the environment
    env = Env()
    state_dim = env.grid_size * env.grid_size + env.num_evacuating_facilities + env.num_receiving_facilities
    action_dim = env.num_actions

    # Initialize the agent
    agent = Actor_Critic_Agent(state_dim, action_dim)

    # Training loop
    num_episodes = 1000
    # max_steps_per_episode = 5000
    total_rewards = []
    time_steps = []
    # trajectories = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        # trajectory = [env.vehicle_position]

        while not done:
            # mask = agent.mask(action_dim, env.vehicle_position)
            action, log_probs, values = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(log_probs=log_probs, values=values, rewards=[reward])
            
            state = next_state
            total_reward += reward
            # trajectory.append(env.vehicle_position.copy())


            # if env.time_step > max_steps_per_episode:
            #     break
                
        # print(state[-8:])
        # print(f"Episode {episode + 1}, Total Reward: {total_reward}, Total Time: {env.time_step}, Total Steps: {iter}")
        # print(trajectory)
                
        total_rewards.append(total_reward)
        time_steps.append(env.time_step)
        # trajectories.append(trajectory)

    # print(total_rewards)

    # Output trajectory after training
    # print("Vehicle Trajectory After training:")
    # for episode, episode_trajectory in enumerate(trajectory):
    #     print(f"Episode {episode + 1}: {episode_trajectory}")

    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Return vs. Training Episode')
    plt.show()
    
    plt.plot(time_steps)
    plt.xlabel('Episode')
    plt.ylabel('Time Steps')
    plt.title('Time Steps vs. Training Episode')
    plt.show()
