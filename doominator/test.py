""" Agent testing file.

This file allows users to watch an agent's performance in a particular
ViZDoom environment.

@author: Rodrigo Pereira
"""
import os
import torch
import tianshou as ts
from network import DQN
from pprint import pprint
from vizdoom_env import *
from tianshou.trainer import onpolicy_trainer
from torch.optim.lr_scheduler import LambdaLR
from tianshou.policy import ICMPolicy, PPOPolicy
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.common import ActorCritic 
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

# Using CUDA if it's available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining some policy hyperparameters.
LR = 0.00002
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.5
VF_COEF = 0.5
ENT_COEF = 0.01

# Defining the training config.
BATCH_SIZE = 64
BUFFER_SIZE = 100000
STEP_PER_EPOCH = 100000
STEP_PER_COLLECT = 1000
REPEAT_PER_COLLECT = 4
TEST_NUM = 1


def dist(p):
    """
    This function defines the distribution function.
    """
    return torch.distributions.Categorical(logits=p)


def watch(env_name, n_epochs):
    """
    This function allows an user to watch the agent's
    performance.
    """
    # Creating the model path.
    save_path = f"doominator/log/{env_name}_ppo_icm"

    try:
        os.mkdir(save_path)
    
    except FileExistsError:
        pass

    # Initializing the environment and the agent.
    env, test_envs = make_vizdoom_env(
        task=env_name, 
        frame_skip=4, 
        res=(4, 84, 84), 
        save_lmp=False,
        seed=42, 
        training_num=0,
        test_num=TEST_NUM
    )

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape or env.action_space.n

    # Initializing the actor, critic and intrinsic curiosity networks.
    net = DQN(*state_shape, action_shape).to(device)
    feature_net = DQN(*state_shape, action_shape).to(device)

    actor = Actor(net, action_shape, device=device, softmax_output=False)
    critic = Critic(net, device=device)

    icm_net = IntrinsicCuriosityModule(
        feature_net=feature_net.net,
        feature_dim=feature_net.output_dim,
        action_dim=np.prod(action_shape), 
        device=device
    )

    optimizer = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=LR)
    icm_optimizer = torch.optim.Adam(icm_net.parameters(), lr=LR)

    # Creating the PPO policy.
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        dist_fn=dist
    ).to(device)

    # Creating the ICM policy
    policy = ICMPolicy(
        policy=policy,
        model=icm_net,
        optim=icm_optimizer,
        lr_scale=0.001,
        reward_scale=0.01,
        forward_loss_weight=0.2
    ).to(device)

    # Verifying if the agent's model file exists.
    try:
        policy.load_state_dict(torch.load(os.path.join(save_path, "policy.pth"), map_location=device))

        print(f"Loaded model from {os.path.join(save_path, 'policy.pth')}, putting it into eval mode.")

        policy.eval()

    except FileNotFoundError:
        raise FileNotFoundError

    # Initializing a replay buffer.
    buffer = VectorReplayBuffer(
        total_size=BUFFER_SIZE,
        buffer_num=len(test_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=4
    )

    # Creating a testing collector.
    test_collector = Collector(policy, test_envs, buffer, exploration_noise=True)

    # Testing the policy.
    result = test_collector.collect(n_episode=5)

    rew = result["rews"].mean()
    lens = result["lens"].mean() * 4

    print(f"Mean reward (over {result['n/ep']} episodes): {rew}")
    print(f"Mean length (over {result['n/ep']} episodes): {lens}")


if __name__ == "__main__":
    name = str(input("Enter the name of the environment: "))

    # Watching the agent.
    watch(name, n_epochs=6)