# Cartpole-Game-DQN

A Reinforcement Learning project that solves the classic **CartPole** environment using a **Deep Q-Network (DQN)**. This repository contains the training script and a pre-trained model to balance the pole on the cart.

## 🎮 Features
- **Deep Q-Network (DQN)** implementation.
- **Dual Modes:** Train a new model or Play with a pre-trained one.
- **Pre-trained Model:** Includes `dqn_cartpole.pth` for immediate demonstration.
- **Customizable:** Easy to adjust episodes and training parameters.

---

## 📂 Project Structure


Cartpole-Game-DQN/
│
├── dqnv2.py              # Main script for training and testing
├── dqn_cartpole.pth      # Pre-trained model weights
└── README.md             # Project documentation

## ⚙️ Setup & Installation
# Prerequisites

Make sure you have Python installed. You will need the following libraries:
Bash

pip install gym torch numpy

# Cloning the Repository
Bash

git clone [https://github.com/Ayushyaverma/Cartpole-Game-DQN.git](https://github.com/Ayushyaverma/Cartpole-Game-DQN.git)
cd Cartpole-Game-DQN

## 🚀 Usage

The main logic is handled in dqnv2.py. You can switch between Training Mode and Playing Mode by modifying the TRAIN variable inside the code.
# 1. Play with Pre-trained Model (Default)

To watch the agent play using the saved model (dqn_cartpole.pth):

    Open dqnv2.py.

    Set the TRAIN variable to False:
    Python

TRAIN = False

Run the script:
Bash

    python dqnv2.py

# 2. Train a New Model

To train the agent from scratch:

    Open dqnv2.py.

    Set the TRAIN variable to True:
    Python

TRAIN = True

Run the script:
Bash

    python dqnv2.py

    The model will train and save the weights to dqn_cartpole.pth upon completion.

# 3. Adjusting Episodes

You can control how many times the agent plays or trains by changing the episodes variable in the script.
Python

episodes = 1000  # Set this to your desired number

## 🧠 How It Works

    Training: The agent explores the environment using an epsilon-greedy strategy, storing experiences in a replay buffer and learning to maximize the reward (keeping the pole upright).

    Testing: The agent uses the learned policy (saved in .pth) to make optimal decisions without exploration.

## 🤝 Contributing

Feel free to fork this repository and experiment with different hyperparameters or network architectures!

    Fork the repo.

    Create your feature branch.

    Commit your changes.

    Push to the branch.

    Create a Pull Request.
