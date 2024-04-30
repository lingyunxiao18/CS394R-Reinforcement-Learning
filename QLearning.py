import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Define Evacuation Environment class
class EvacuationEnvironment:
    def __init__(self, facility_data, distance_data, vehicle_data):
        self.Number_of_ranks=4
        self.current_pair=None
        self.rank = 0
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
        self.current_pair = self.vehicle_location
        return self._get_state()

    def step(self, action,rank):
        self.current_pair=(evacuating_facility, receiving_facility) = self.pairs[action]
        self.rank=rank
        
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
            
        if rank == env.Number_of_ranks:
            reward -= self.distance[(self.vehicle_location, self.initial_location)]
            self.vehicle_location = self.initial_location
            if done:
                reward += 1000  #  bonus for completing all demand and returning to staging area
                print(f'RETURNED TO STAGING AREA {self.vehicle_location}')
            else:
                reward -= 1000
                print('!Incomplete Evacuation!')
            
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        return (self.rank, self.current_pair)
    
# In[2]: 

class QLearningAgent:
    def __init__(self, state_dim, action_dim, alpha=0.5, gamma=0.99, epsilon=0.1):  #alpha=0.1, gamma=0.99, epsilon=0.2):
        self.Q = {(0, i): np.zeros(action_dim) for i in env.I}  # Initialize Q-table as a dictionary
        self.Q.update({(rank, (j,k)): np.zeros(action_dim) for rank in range(1, env.Number_of_ranks + 1) for j, k in env.pairs})
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy
        self.action_dim = action_dim
        self.prev_action = None
        self.prev_state = None

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_dim)  # Random action
        else:
            action = np.argmax(self.Q[state])   # Greedy action
            print(f'greedy action {action} chosen')
        self.prev_action = action
        self.prev_state = state
        return action
            
    def update(self, reward, next_state):
        if self.prev_state is not None and self.prev_action is not None:
            max_next_q_value = np.max(self.Q[next_state])  # Maximum Q-value for the next state
            td_target = reward + self.gamma * max_next_q_value
            td_error = td_target - self.Q[self.prev_state][self.prev_action]
            self.Q[self.prev_state][self.prev_action] += self.alpha * td_error


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
       
    state_dim = env.Number_of_ranks * len(env.pairs) + len(env.I)
    action_dim = len(env.pairs)
    
    # Initialize agent
    agent = QLearningAgent(state_dim=state_dim, action_dim=action_dim)
       
    # Training loop
    num_episodes = 100000
    total_rewards = []
    
    for episode in range(num_episodes):
        env.rank = 0
        if np.random.rand() < agent.epsilon:
            starting_staging_area = np.random.choice(env.I)  # Random action
        else:
            max_value=0
            for i in env.I:
                value=np.max(agent.Q[(0,i)])
                if value>=max_value:
                    max_value=value
                    starting_staging_area = i

        state = env.reset(starting_staging_area)
        total_reward = 0
        done = False

        #while not done:
        for rank in range(env.Number_of_ranks):          
            env.rank=rank
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action,rank+1)
            agent.update(reward, next_state)
            state = next_state
            total_reward += reward
        print(f"Episode {episode + 1}, Total Reward: {total_reward}\n")
        total_rewards.append(total_reward)
        
    print(agent.Q)
    print("\n")
    for state in agent.Q:
        greedy_action = np.argmax(agent.Q[state])
        print(f"State: {state}, Greedy Action: {env.pairs[greedy_action]}, with Q= {agent.Q[state][greedy_action]}")
                
    
    # Plotting the data
    plt.figure(figsize=(12, 6))
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards vs. Episode of Training')
    plt.show()

print("\nElapsed time:", time.time() - start_time)


# In[2]:
#Hyperparameter tuning   
facility_data = pd.read_csv('facility.csv')
distance_data = pd.read_csv('distance.csv')
vehicle_data = pd.read_csv('vehicle.csv')

# Initialize environment
env = EvacuationEnvironment(facility_data, distance_data, vehicle_data)
   
state_dim = env.Number_of_ranks * len(env.pairs) + len(env.I)
action_dim = len(env.pairs)

num_episodes = 100000    
# Hyperparameter tuning
alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
gammas = [0.99] #[0.9,0.95,0.99]
epsilons = [0.2] #[0.1,0.3,0.5,0.7,0.9]

best_reward = float('-inf')
best_hyperparameters = None
avg_reward_array=[]
for alpha in alphas:   
    for gamma in gammas:
        for epsilon in epsilons:
            agent = QLearningAgent(state_dim=state_dim, action_dim=action_dim, alpha=alpha, gamma=gamma, epsilon=epsilon)
            total_rewards = []
            for episode in range(num_episodes):
                env.rank = 0
                state = env.reset()
                total_reward = 0
                done = False

                for rank in range(env.Number_of_ranks):          
                    env.rank = rank
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action, rank+1)
                    agent.update(reward, next_state)
                    state = next_state
                    total_reward += reward

                total_rewards.append(total_reward)
                
                
            avg_reward = np.mean(total_rewards)
            avg_reward_array.append(avg_reward)
            
            print(f'Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}, Average Reward: {avg_reward}')

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_hyperparameters = (alpha, gamma, epsilon)

            
plt.figure(figsize=(12, 6))
plt.plot(alphas, avg_reward_array, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Average Reward')
#plt.title('Average Reward vs. Alpha')
plt.grid(True)
plt.show()

print(f'Best hyperparameters: {best_hyperparameters}, Best average reward: {best_reward}')
