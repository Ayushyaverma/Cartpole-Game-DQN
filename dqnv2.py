import gymnasium as gym
import random
import math
from collections import deque, namedtuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
TRAIN = True   # change to True only when you want to retrain


# -----------------------------
# Replay Memory
# -----------------------------
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -----------------------------
# DQN Network
# -----------------------------
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Epsilon-greedy policy
# -----------------------------
def select_action(state, steps_done, policy_net, n_actions):
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000   # SLOW decay (important)

    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1.0 * steps_done / eps_decay)

    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

# -----------------------------
# Optimization step
# -----------------------------
def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_values = policy_net(state_batch).gather(1, action_batch)

    non_final_mask = torch.tensor(
        [s is not None for s in batch.next_state],
        dtype=torch.bool
    )

    next_state_values = torch.zeros(batch_size)

    if non_final_mask.any():
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1)[0].detach()
        )

    expected_q_values = reward_batch + gamma * next_state_values

    loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =========================================================
# TRAINING
# =========================================================
env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(obs_dim, n_actions)
target_net = DQN(obs_dim, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
memory = ReplayMemory(10000)

num_episodes = 1500
batch_size = 64
gamma = 0.99
steps_done = 0

print("Training started...")

if TRAIN:
    print("Training started...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)

        episode_reward = 0

        for t in range(1000):
            action = select_action(state, steps_done, policy_net, n_actions)
            steps_done += 1

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward

            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            if terminated:
               next_state_tensor = None
            else:
                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)

            memory.push(state, action, next_state_tensor, reward_tensor)
            state = next_state_tensor if next_state_tensor is not None else state

            optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)

            if terminated or truncated:
                break

        # Soft update target network
        tau = 0.005
        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.copy_(tau * pp.data + (1 - tau) * tp.data)

        if episode % 50 == 0:
            print(f"Episode {episode}, reward = {episode_reward}")

    torch.save(policy_net.state_dict(), "dqn_cartpole.pth")
    env.close()
    print("Training complete")
else:
    policy_net.load_state_dict(torch.load("dqn_cartpole.pth"))
    policy_net.eval()
    print("Loaded trained model")


# =========================================================
# VISUAL TEST
# =========================================================
test_env = gym.make("CartPole-v1", render_mode="human")

state, _ = test_env.reset()
state = torch.from_numpy(state).float().unsqueeze(0)

total_reward = 0

for _ in range(500):
    test_env.render()

    with torch.no_grad():
        action = policy_net(state).argmax(dim=1).item()

    next_state, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward

    if terminated or truncated:
        break

    state = torch.from_numpy(next_state).float().unsqueeze(0)
    time.sleep(0.02)

print("Test episode reward:", total_reward)
time.sleep(5)
test_env.close()
