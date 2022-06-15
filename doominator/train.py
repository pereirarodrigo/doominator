""" Agent training file.

This file contains the implementation of the PPO algorithm and the training
procedure. The resulting model is saved in the "models" folder.

@author: Rodrigo Pereira
"""
import os
import torch
import tianshou as ts
from network import CNN
from pprint import pprint
from vizdoom_env import *
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.common import ActorCritic 
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import Collector, VectorReplayBuffer

# Using CUDA if it's available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining some policy hyperparameters.
LR = 1e-3
GAMMA = 0.99

# Defining the training config.
BATCH_SIZE = 32
BUFFER_SIZE = 20000
STEP_PER_EPOCH = 10000
STEP_PER_COLLECT = 10
REPEAT_PER_COLLECT = 4
TRAIN_NUM = 5
TEST_NUM = 50
LOGGER = ts.utils.TensorboardLogger(SummaryWriter("doominator/log/ppo_clip"))


def dist(p):
    """
    This function defines the distribution function.
    """
    return torch.distributions.Categorical(logits=p)


def save_best_fn(policy, model_path):
    """
    This function saves the best model.
    """
    torch.save(policy.state_dict(), os.path.join(model_path, "policy.pth"))


def train(env_name, n_epochs):
    """
    This function trains the agent.
    """
    # Creating the model path.
    save_path = f"doominator/models/{env_name}_ppo_clip"

    try:
        os.mkdir(save_path)
    
    except FileExistsError:
        pass

    # Initializing the environment and the agent.
    env, train_envs, test_envs = make_vizdoom_env(
        task=env_name, 
        frame_skip=4, 
        res=(4, 84, 84), 
        save_lmp=False,
        seed=42, 
        training_num=TRAIN_NUM,
        test_num=TEST_NUM
    )

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape or env.action_space.n

    # Initializing the actor and critic networks.
    net = CNN(*state_shape, action_shape).to(device)

    actor = Actor(net, action_shape, device=device, softmax_output=False)
    critic = Critic(net, device=device)

    optimizer = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=LR)

    # Creating the agent.
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        dist_fn=dist,
        discount_factor=GAMMA
    ).to(device)

    # Verifying if the model already exists, so that the training process 
    # can continue.
    try:
        policy.load_state_dict(torch.load(os.path.join(save_path, "policy.pth"), map_location=device))

    except FileNotFoundError:
        pass

    # Initializing a replay buffer.
    buffer = VectorReplayBuffer(
        total_size=BUFFER_SIZE,
        buffer_num=TRAIN_NUM,
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=4
    )

    # Creating training and testing collectors.
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    # Training the policy.
    train_collector.collect(n_step=BATCH_SIZE * TRAIN_NUM)

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        n_epochs,
        STEP_PER_EPOCH,
        REPEAT_PER_COLLECT,
        TEST_NUM,
        BATCH_SIZE,
        STEP_PER_COLLECT,
        save_best_fn=save_best_fn(policy, save_path),
        logger=LOGGER,
        test_in_train=False
    )

    pprint(result)


if __name__ == "__main__":
    name = str(input("Enter the name of the environment: "))

    # Training the agent.
    train(name, n_epochs=100)

