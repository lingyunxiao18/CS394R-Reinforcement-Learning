import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Define Evacuation Environment class
class EvacuationEnvironment:
    def __init__(self, facility_data, distance_data, vehicle_data):
        self.I = facility_data[facility_data['CATEGORY'] == 'Staging Area']['Facility'].tolist()
        self.J = facility_data[facility_data['CATEGORY'] == 'Evacuating Hospital']['Facility'].tolist()
        self.K = facility_data[facility_data['CATEGORY'] == 'Receiving Hospital']['Facility'].tolist()
        self.pairs = [(j, k) for j in self.J for k in self.K]
        self.V = vehicle_data['Vehicle'].tolist()
        
        self.Identifier = {row['Facility']: row['Identifier'] for _, row in facility_data.iterrows()}
        self.D = {row['Facility']: row['BEDS'] for _, row in facility_data.iterrows() if row['CATEGORY'] == 'Evacuating Hospital'}
        self.C = {row['Facility']: row['BEDS'] for _, row in facility_data.iterrows() if row['CATEGORY'] == 'Receiving Hospital'}

        self.initial_demand = {j: self.D[j] for j in self.J}
        self.initial_capacity = {k: self.C[k] for k in self.K}

        # Assuming distance data contains distances between facilities
        self.distance = {(row['From-to'].split(',')[0], row['From-to'].split(',')[1]): row['Distance'] for _, row in distance_data.iterrows()}

    def reset(self, starting_staging_area=None):
        self.demand = self.initial_demand.copy()
        self.capacity = self.initial_capacity.copy()
        
        if starting_staging_area is None:
            self.vehicle_location = random.choice(self.I)  # Randomly select a staging area
        else:
            self.vehicle_location = starting_staging_area
            
        self.initial_location = self.vehicle_location
        return self._get_state()

    def step(self, action):
        evacuating_facility, receiving_facility = self.pairs[action]
        print('next state: evacuating_facility, receiving_facility = ',evacuating_facility, receiving_facility)
        reward = -self.distance[(self.vehicle_location, evacuating_facility)]
        #print('self.distance[(self.vehicle_location, evacuating_facility)] = ', reward)
        
        if self.demand[evacuating_facility] <= self.capacity[receiving_facility]:
            evacuated = self.demand[evacuating_facility]
        else:
            evacuated = self.capacity[receiving_facility]
        
        print('evacuated = ',evacuated)
        self.demand[evacuating_facility] -= evacuated
        self.capacity[receiving_facility] -= evacuated
        reward -= (2 * evacuated - 1) * self.distance[(evacuating_facility, receiving_facility)]
        #print('reward evacuation distance added= ',reward)
        if evacuated > 0:
            reward += 100 * evacuated  # Small bonus for saving evacuees
         #   print('reward + evacuated added= ',reward)
        elif evacuated == 0:
            reward -= 2 * self.distance[(evacuating_facility, receiving_facility)] * sum(self.demand.values())
          #  print('reward -2 added= ',reward)
            
        self.vehicle_location = receiving_facility
        done = all(d == 0 for d in self.demand.values())
        if done:
            reward -= self.distance[(self.vehicle_location, self.initial_location)]
            self.vehicle_location = self.initial_location
            reward += 1000  #  bonus for completing all demand and returning to staging area
            print(f'RETURNED TO STAGING AREA {self.vehicle_location}')
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        demand_capacity_state = np.array(list(self.demand.values()) + list(self.capacity.values()), dtype=np.int32)
        vehicle_loc_state = np.array([int(self.Identifier[self.vehicle_location])], dtype=np.int32)
        return np.concatenate([demand_capacity_state, vehicle_loc_state])

    
# In[2]: 
    
# Define REINFORCE Agent class
class REINFORCE_Agent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=0.001, gamma=1, max_grad_norm=0.5):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.prev_action = None  # Attribute to store the previous action

    def select_action(self, state, unavailable_actions):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        
        # Exclude the previous action from the available actions
        if self.prev_action is not None:
            probs[0, self.prev_action] = 0
            probs /= probs.sum()  # Normalize probabilities
            
            # Filter out action pairs with evacuating facility with 0 remaining demand
        for i in range(len(env.pairs)):
            evacuating_facility, receiving_facility = env.pairs[i]
            if (evacuating_facility, receiving_facility) in unavailable_actions:
                probs[0, i] = 0
    
        # Normalize probabilities
        probs /= probs.sum()
            
        action = torch.multinomial(probs, 1) #tensor([[2]]) this is index so the 3rd element in the action space
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action) # log_prob = tensor([[-1.9080]], grad_fn=<GatherBackward0>)
        self.prev_action = action.item()  # Update previous action 
        return action.item(), log_prob # action.item()= 2

    def update(self, rewards, log_probs):
        discounted_returns = self.calculate_discounted_returns(rewards)
        policy_loss = []
        for log_prob, Gt in zip(log_probs, discounted_returns):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def calculate_discounted_returns(self, rewards):
        discounted_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_returns.insert(0, R)
        return torch.tensor(discounted_returns)

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
    
# In[2]: 
#####################################################################################################################################
#For Evacaution environment
if __name__ == "__main__":
    # Read facility, distance, and vehicle data from CSV
    facility_data = pd.read_csv('facility.csv')
    distance_data = pd.read_csv('distance.csv')
    vehicle_data = pd.read_csv('vehicle.csv')
    
    # Initialize environment
    env = EvacuationEnvironment(facility_data, distance_data, vehicle_data)
    
    # Example usage
    #state = env.reset()
    #print("Initial state:", state)   #Initial state: ['36.0' '29.0' '16.0' '37.0' '22.0' '44.0' '43.0' 's1']
    
    state_dim = len(env.reset())
    action_dim = len(env.pairs)
#####################################################################################################################################
    # Initialize agent
    agent = REINFORCE_Agent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, lr=0.001, gamma=1)

    # Training loop
    num_episodes = 1000
    total_rewards = []
    #highest possible reward for optimal route s1-> e2->r2 (29) ->e1->r5( 36) ->s1
    best_reward = -env.distance[('s1', 'e2')] - (2 * 29 - 1) * env.distance[('e2', 'r2')] + 100*29 -env.distance[('r2', 'e1')] - (2 * 36 - 1) * env.distance[('e1', 'r5')] + 100*36 + 1000
    highest_reward=0
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        log_probs = []

        while not done:
            print("\nstate=",state)
            unavailable_actions = [(j, k) for j, k in env.pairs if env.demand[j] == 0]
            action, log_prob = agent.select_action(state, unavailable_actions)

            #action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob.detach().requires_grad_())
            agent.update(rewards=[reward], log_probs=log_probs)
            state = next_state
            total_reward += reward

            if total_reward >= highest_reward:
                highest_reward_episode_info = {
                                            'next_state': state,
                                            'evacuated_info': env.initial_demand
                                            }
                highest_reward = total_reward
                        
        print(f"Episode {episode + 1}, Total Reward: {total_reward}\n")
        total_rewards.append(total_reward)
        
    highest_reward_reached = (highest_reward>=best_reward)
    print('highest_reward_reached = ', highest_reward_reached)
    print('\nhighest_reward_reached = ', highest_reward)
    print('\nhighest_reward_episode_info = ', highest_reward_episode_info) 
    
    # Plotting the data
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards vs. Episode of Training')
    plt.show()
    
    