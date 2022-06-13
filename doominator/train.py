""" Agent training file.

This file contains the implementation of the PPO algorithm and the training
procedure. The resulting model is saved in the "models" folder.

@author: Rodrigo Pereira
"""
import os
import torch
import vizdoom
import itertools as it
from policy import PPOPolicy
from file_utils import create_game

# Using CUDA if it's available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg_path, cfg_name, n_timesteps):
    """
    This function trains the agent.
    """
    # Initializing the environment and the agent.
    env = create_game(cfg_path)
    n = env.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    agent = PPOPolicy(
        env=env, 
        action_dim=len(actions),
        batch_size=64
    ).to(device)

    # Training the agent.
    agent.learn(n_timesteps)

    # Saving the trained agent.
    torch.save(agent.actor.state_dict(), f"models/{cfg_name}_agent/actor.pt")
    torch.save(agent.critic.state_dict(), f"models/{cfg_name}_agent/critic.pt")


if __name__ == "__main__":
    name = str(input("Enter the name of the environment: "))
    cfg_path = os.path.join(vizdoom.scenarios_path, f"{name}.cfg")

    # Training the agent.
    train(cfg_path, name, n_timesteps=10)

