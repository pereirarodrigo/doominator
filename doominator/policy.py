""" Policy implementation.

This file implements the policy used by Doominator, the PPO algorithm
(read more about PPO here: https://arxiv.org/pdf/1707.06347). More
specifically, the variant used here is PPO-Clip, which does not have
a KL-divergence (something like a measure of the distance between two 
probability distributions) term in the objective and has no constraint.

@author: Rodrigo Pereira
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import CNN
from tqdm import trange
from file_utils import preprocess

class PPOPolicy(nn.Module):
    """
    PPO-clip implementation, heavily based on the PPO policy implemented in:
    
    - Stable Baselines 3 
    - https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8.
    
    The overall training architecture had to be changed to fit the structure of ViZDoom, which does
    not use methods such as OpenAI Gym's env.reset() and env.step(action) (unless you use an env 
    wrapper).
    """
    def __init__(self, env, action_dim, batch_size):
        super(PPOPolicy, self).__init__()
        # Calling the init_hyperparamters function to initialize the hyperparameters.
        self.init_hyperparameters()

        # Training parameters.
        self.env = env
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Defining the actor and critic networks.
        self.actor = CNN(output_size=self.action_dim).to(self.device)
        self.critic = CNN(output_size=1).to(self.device)

        # Defining the actor and critic optimizers.
        self.actor_optim = optim.SGD(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = optim.SGD(self.critic.parameters(), lr=self.learning_rate)

        # Initializing a covariance matrix for the exploration process.
        self.cov_var = torch.full(size=(self.action_dim, ), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        

    def init_hyperparameters(self):
        # Initializes the policy's hyperparameters.
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.max_timesteps_per_batch = 128
        self.max_timesteps_per_episode = 1
        self.n_updates_per_iteration = 5
        self.clip = 0.2


    def get_action(self, obs):
        """
        This function returns the action to be taken given the observation.
        """
        # Querying the actor for a mean action.
        obs = np.expand_dims(obs, axis=0)
        mean = self.actor(obs)

        # Creating a multivariate normal distribution.
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)

        # Sampling an action from the distribution and getting its log prob.
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Returning the sampled action and log prob of that action.
        # We use "detach()" to get rid of the computation graph.
        return torch.flatten(action).cpu().detach().numpy(), log_prob.detach()


    def compute_rtgs(self, batch_rews):
        """
        Computes a batch of rewards-to-go.
        """
        # Initializing the rewards-to-go.
        batch_rtgs = []

        # Iterating over the batch of rewards.
        # Notably, we iterate backwards.
        for rewards in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(rewards):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Converting the batch rewards-to-go to a tensor.
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)

        return batch_rtgs


    def rollout(self):
        """
        This function collects data from a set of episodes by running the actor policy
        """
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_tg = []
        batch_lens = []

        # Initializing the episode rewards.
        ep_rewards = []

        for ep in trange(self.max_timesteps_per_batch):
            self.env.new_episode()

            ep_rewards = []

            # Initializing our observations.
            obs = preprocess(self.env.get_state().screen_buffer) 

            for ep_t in range(self.max_timesteps_per_episode):
                batch_obs.append(obs)

                # Computing the action, getting observations and computing rewards.
                action, log_prob = self.get_action(obs)
                #obs = preprocess(self.env.get_state().screen_buffer) 
                reward = self.env.make_action(action, 12)
                done = self.env.is_episode_finished()
                
                # Collecting the actions, rewards and log probabilities.
                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    self.env.new_episode()

            # Collects the episodic length and rewards.
            batch_lens.append(ep_t + 1)    # as timestep starts at 0
            batch_rewards.append(ep_rewards)

        # Reshaping our data as tensors.
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=self.device)

        # Calculating the batch rewards-to-go.
        batch_rewards_tg = self.compute_rtgs(batch_rewards).to(self.device)

        return batch_obs, batch_acts, batch_log_probs, batch_rewards_tg, batch_lens


    def evaluate(self, batch_obs, batch_acts):
        """
        Calculates the predicted values based on the provided Q-values.
        """
        # Querying the critic network for a mean value for each 
        # observation in the batch.
        V = self.critic(batch_obs).squeeze()

        # Like when getting an action, we query the actor, create a 
        # multivariate normal distribution and return its log prob as well.
        mean = self.actor(batch_obs)

        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(batch_acts)

        return V, log_prob


    def learn(self, total_timesteps:int):
        """
        Function that performs the learning of the policy given a number of
        timesteps.
        """
        t_so_far = 0
        it_so_far = 0

        for t in range(total_timesteps):
            it_so_far += 1

            # Sample a batch of experiences.
            obs, acts, log_probs, rewards_tg, lens = self.rollout()

            t_so_far += np.sum(lens)

            # Computing the predicted values.
            V, _ = self.evaluate(obs, acts)

            # Calculating the advantage function.
            A_k = rewards_tg - V.detach()    # once again removing the computation graph

            # Normalizing the advantage.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)  # 1e-10 to avoid division by 0

            for _ in range(self.n_updates_per_iteration):
                # Computing pi_theta (a_t | s_t)
                V, curr_log_probs = self.evaluate(obs, acts)

                # Calculating ratios.
                ratios = torch.exp(curr_log_probs - log_probs)

                # Calculating surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculating the actor and critic losses.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(V, rewards_tg)

                # Calculating gradients and performing backward propagation for the actor.
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculating gradients and performing backward propagation for the critic.
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            # Printing the results
            print(f"\n| Iteration {it_so_far} results:")
            print(f"|-----------------------------")
            print(f"| Avg. reward: {torch.mean(rewards_tg):.2f}")
            print(f"| Max reward: {torch.max(rewards_tg):.2f}")
            print(f"| Min reward: {torch.min(rewards_tg):.2f}")
            print(f"| Timesteps: {t_so_far}\n")
