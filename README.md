# CS394R-Reinforcement-Learning
Final Project of CS 394R: Reinforcement Learning, UT Austin

This repository contains a Python implementation of various reinforcement learning (RL) agents that are designed to navigate and evacuate individuals in a simulated environment with evacuating and receiving facilities. The RL agent is trained using discrete (Sarsa, Q-Learning, Expected Sarsa) and continuous algorithms (REINFORCE, Actor-Critic) to maximize cumulative rewards by efficiently transferring evacuees from evacuating to receiving facilities. 

## Overview

The core idea of this project is to demonstrate how an RL agent can learn to navigate an evacuation scenario by picking up evacuees from evacuating facilities and dropping them off at receiving facilities. The agent's goal is to maximize the total return within a given environment.

## Usage

### Setup

1. **Clone Repository**:

   ```bash
   git clone https://github.com/lingyunxiao18/CS394R-Reinforcement-Learning.git
   cd CS394R-Reinforcement-Learning

2. **Run code**

   ```bash
   python3 Evacuation_REINFORCE.py
   python3 Evacuation_ActorCritic.py

