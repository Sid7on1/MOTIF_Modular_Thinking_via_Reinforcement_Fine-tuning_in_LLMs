import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Simplified example: Simulating modular reasoning with a small LLM and RL finetuning

class SimpleLLM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLLM, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1) # Output probabilities for actions

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.softmax(x)

def run_episode(model, initial_state, num_modules=3):
    """Simulates an episode of modular reasoning."""
    state = initial_state
    log_probs = []
    rewards = []
    actions = []

    for _ in range(num_modules):
        # 1. Get action probabilities from the LLM
        probs = model(torch.tensor(state, dtype=torch.float32))
        m = Categorical(probs)
        action = m.sample()
        log_probs.append(m.log_prob(action))
        actions.append(action.item())

        # 2. Simulate environment interaction (simplified reward function)
        # In a real scenario, this would involve executing the module and observing the next state.
        reward = 0.1  # Small reward for each step
        if action.item() == 1: # Simulate a "correct" action
            reward = 1.0 # Larger reward for correct action
        rewards.append(reward)

        # 3. Update state (simplified)
        state = state + np.random.normal(0, 0.1, size=len(state)) # Add some noise

    # Final reward (simplified)
    final_reward = 0.0
    if actions[-1] == 1: # If the last action was "correct"
        final_reward = 5.0
    rewards[-1] += final_reward

    return log_probs, rewards

def reinforce(model, optimizer, num_episodes=100, initial_state_size=10):
    """Implements the REINFORCE algorithm for finetuning the LLM."""
    gamma = 0.99 # Discount factor

    for episode in range(num_episodes):
        # 1. Generate an episode
        initial_state = np.random.rand(initial_state_size)
        log_probs, rewards = run_episode(model, initial_state)

        # 2. Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8) # Normalize

        # 3. Calculate loss and update model
        loss = 0
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss += -log_prob * reward # REINFORCE loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss.item()}")

if __name__ == '__main__':
    # Example usage
    input_size = 10
    hidden_size = 20
    output_size = 2  # 2 possible actions (e.g., different modules to use)

    model = SimpleLLM(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    reinforce(model, optimizer, num_episodes=200)

    # Example of using the finetuned model
    initial_state = np.random.rand(input_size)
    probs = model(torch.tensor(initial_state, dtype=torch.float32))
    print("Action probabilities after finetuning:", probs)