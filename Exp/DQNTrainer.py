import math
import os
import random
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from spikingjelly.activation_based import functional
from torch.utils.tensorboard import SummaryWriter

from algorithm.DQN import ReplayMemory


class DQNTrainer:
    def __init__(
        self,
        env_name: str,
        model_class,
        hidden_size: int,
        t: int,
        use_cuda: bool,
        config: dict,
        seed: int = 1,
    ):

        random.seed(seed)
        np.random.seed(seed)

        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.env = gym.make(env_name).unwrapped

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )

        self.policy_net = model_class(self.n_states, hidden_size, self.n_actions, t).to(
            self.device
        )
        self.target_net = model_class(self.n_states, hidden_size, self.n_actions, t).to(
            self.device
        )

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.memory = ReplayMemory(config["MEMORY_CAPACITY"])

        self.config = config

        self.seed = seed

        self.steps_done = 0

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        eps_threshold = self.config["EPS_END"] + (
            self.config["EPS_START"] - self.config["EPS_END"]
        ) * math.exp(-1.0 * self.steps_done / self.config["EPS_DECAY"])

        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)
                functional.reset_net(self.policy_net)
                return action
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.config["BATCH_SIZE"]:
            return

        transitions = self.memory.sample(self.config["BATCH_SIZE"])
        batch = self.transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.config["BATCH_SIZE"], device=self.device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        functional.reset_net(self.target_net)

        expected_state_action_values = reward_batch + (
            self.config["GAMMA"] * next_state_values
        )
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        functional.reset_net(self.policy_net)
        return loss.item()

    def train(
        self,
        num_episodes: int,
        model_dir: str,
        log_dir: str,
        log_name: str,
        model_name: str,
    ):
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        writer = SummaryWriter(log_dir=log_dir)
        max_reward = 0

        for i_episode in range(num_episodes):
            self.env.reset(seed=self.seed)
            state = torch.zeros(
                [1, self.n_states], dtype=torch.float, device=self.device
            )
            total_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )
                total_reward += reward
                next_state = (
                    torch.from_numpy(next_state).float().to(self.device).unsqueeze(0)
                )
                reward = torch.tensor([reward], device=self.device)

                if terminated or truncated:
                    next_state = None

                self.memory.push(state, action, next_state, reward)
                state = next_state

                loss = self.optimize_model()

                if terminated or truncated:
                    # print(f"Episode: {i_episode}, Reward: {total_reward}")
                    # print(f"Loss: {loss}")
                    if loss is not None:
                        writer.add_scalar(f"{log_name}-Loss", loss, i_episode)

                    writer.add_scalar(f"{log_name}-Reward", total_reward, i_episode)
                    if total_reward > max_reward:
                        max_reward = total_reward
                        torch.save(
                            self.policy_net.state_dict(),
                            os.path.join(model_dir, f"{model_name}_best_model.pt"),
                        )
                        print(f"max_reward={max_reward}, save models")
                    break

            if i_episode % self.config["TARGET_UPDATE"] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        writer.close()
        torch.save(
            self.policy_net.state_dict(),
            os.path.join(model_dir, f"{model_name}_final_model.pt"),
        )
