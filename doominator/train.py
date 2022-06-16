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
from tianshou.trainer import onpolicy_trainer
from tianshou.policy import ICMPolicy, PPOPolicy
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.common import ActorCritic 
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

# Using CUDA if it's available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining some policy hyperparameters.
LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_GRAD_NORM = 0.5
VF_COEF = 0.5
ENT_COEF = 0.01

# Defining the training config.
BATCH_SIZE = 32
BUFFER_SIZE = 20000
STEP_PER_EPOCH = 100000
STEP_PER_COLLECT = 10
REPEAT_PER_COLLECT = 4
TRAIN_NUM = 5
TEST_NUM = 50


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

    # Initializing the actor, critic and intrinsic curiosity networks.
    net = CNN(*state_shape, action_shape).to(device)
    feature_net = CNN(*state_shape, action_shape).to(device)

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
        dist_fn=dist,
        discount_factor=GAMMA, 
        gae_lambda=GAE_LAMBDA,
        max_grad_norm=MAX_GRAD_NORM,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        reward_normalization=False,
        action_scaling=False,
        action_space=env.action_space,
        eps_clip=0.2,
        value_clip=0,
        dual_clip=None,
        advantage_normalization=1,
        recompute_advantage=0,
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

    # Verifying if we can resume training.
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


    def stop_fn(mean_rewards):
        """
        This function stops the training if the mean reward is greater than or equal
        to the reward threshold.
        """
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
            
        else:
            return False


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
        stop_fn=stop_fn,
        logger=LOGGER,
        test_in_train=False
    )

    pprint(result)


if __name__ == "__main__":
    name = str(input("Enter the name of the environment: "))

    LOGGER = ts.utils.TensorboardLogger(SummaryWriter(f"doominator/log/{name}_ppo_clip"))

    # Training the agent.
    train(name, n_epochs=100)

