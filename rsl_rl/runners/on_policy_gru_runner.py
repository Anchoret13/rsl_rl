import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

from RNN import GRU

class OnPolicyGRURuner:
    def __init__(
            self,
            env:VecEnv,
            train_cfg,
            log_dir = None,
            device = 'cpu'
    ):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.window_size = 50 #NOTE: It is fixed again lol this idiot

        # GRU setup
        self.model_params = {
            "input_size" : env.num_adapt,
            "n_layer"    : 2,
            "output_size": 10,  # TODO: currently it'fixed
            "hidden_size": 150
        }

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        total_actor_input_dim = env.num_obs + self.model_params['output_size']
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        self.actor_critic = actor_critic_class(num_actor_obs = total_actor_input_dim,
                                               num_critic_obs = env.num_privileged_obs,
                                               num_actions = self.env.num_actions, 
                                               **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(self.actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])
        
        
        self.gru = GRU(**self.model_params).to(device)
        self.gru_optimizer = Adam(self.gru.parameters(), lr=0.001)

        # initialize observation buffer
        self.obs_buffer = torch.zeros((env.num_envs, self.window_size, env.observation_space.shape[0]), dtype=torch.float32, device=self.device)
        self.buffer_index = 0

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def update_obs_buffer(self, new_obs):
        assert new_obs.shape == (self.env.num_envs, self.env.num_obs)
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        self.obs_buffer[:, self.buffer_index, :] = new_obs

    def prepare_gru_input(self):
        start_index = (self.buffer_index + 1) % self.window_size
        if start_index + self.window_size <= self.window_size:
            return self.obs_buffer[:, start_index:start_index + self.window_size, :]
        else:
            end_part = self.obs_buffer[:, start_index, :]

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        loss_fn = nn.MSELoss()
        
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    #  GRU training
                    gru_input = self.env.compute_adapt_input().to(self.device)
                    gru_output = self.gru(gru_input)
                    gru_target = self.env.compute_adapt_target().to(self.device) # TODO: double check this
                    gru_loss = loss_fn(gru_output, gru_target)
                    self.gru_optimizer.zero_grad()
                    gru_loss.backward()
                    self.gru_optimizer.step()

                    enhanced_obs = torch.cat((obs, gru_output), dim = 1)
                    actions = self.alg.act(enhanced_obs, critic_obs)
                    



    def log(self):
        pass

    def save(self, path, infos = None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            'gru_state_dict': self.gru.state_dict(),  
        }, path)

    def load(self, path, load_optimizer = True):
        loaded_dict = torch.load(path, map_location = self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.gru.load_state_dict(loaded_dict['gru_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict.get('infos', None)

    def get_inference_policy(self, device = None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)

        def inference_policy(observation):
            self.update_obs_buffer(observation)
            gru_input = self.prepare_gru_input()
            gru_output = self.gru(gru_input)
            enriched_obs = torch.cat((observation, gru_output), dim=1)
            return self.alg.actor_critic.act_inference(enriched_obs)
        
        return inference_policy