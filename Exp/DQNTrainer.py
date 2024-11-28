import math
import os
import random
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from algorithm.DQN import ReplayMemory
from spikingjelly.activation_based import functional
from torch.utils.tensorboard import SummaryWriter


class DQNTrainer:
    """
    DQNTrainer is a class that implements the Deep Q-Learning algorithm for training a neural network to play a given environment.

    Attributes
    -------
        device (torch.device): The device to run the computations on (CPU or GPU).
        env (gym.Env): The environment to train the agent in.
        n_states (int): The number of states in the environment.
        n_actions (int): The number of actions available in the environment.
        transition (namedtuple): A named tuple representing a transition in the environment.
        policy_net (torch.nn.Module): The neural network used to select actions.
        target_net (torch.nn.Module): The neural network used to compute target Q-values.
        optimizer (torch.optim.Optimizer): The optimizer used to update the policy network.
        memory (ReplayMemory): The replay memory to store transitions.
        config (dict): The configuration dictionary containing hyperparameters.
        seed (int): The random seed for reproducibility.
        steps_done (int): The number of steps taken in the environment.

    Methods
    -------
        __init__(env_name: str, model_class, hidden_size: int, t: int, use_cuda: bool, config: dict, seed: int = 1):
            Initializes the DQNTrainer with the given parameters.
        select_action(state: torch.Tensor) -> torch.Tensor:
            Selects an action based on the current state using an epsilon-greedy policy.
        optimize_model():
            Optimizes the policy network by sampling a batch of transitions from the replay memory and performing a gradient descent step.
        train(num_episodes: int, model_dir: str, log_dir: str, log_name: str, model_name: str):
            Trains the agent for a given number of episodes, logs the training progress, and saves the best and final models.
    """

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
        """
        Initialize the DQNTrainer.

        Args
        -------
            env_name (str): The name of the environment to train on.
            model_class: The class of the model to be used for the policy and target networks.
            hidden_size (int): The size of the hidden layers in the model.
            t (int): A parameter for the model (e.g., time steps).
            use_cuda (bool): Whether to use CUDA for computation.
            config (dict): Configuration dictionary containing various parameters.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.

        Attributes
        -------
            device (torch.device): The device to run the computations on (CPU or CUDA).
            env (gym.Env): The environment instance.
            n_states (int): The number of states in the environment's observation space.
            n_actions (int): The number of actions in the environment's action space.
            transition (namedtuple): A named tuple representing a transition in the environment.
            policy_net (torch.nn.Module): The policy network model.
            target_net (torch.nn.Module): The target network model.
            optimizer (torch.optim.Optimizer): The optimizer for the policy network.
            memory (ReplayMemory): The replay memory to store transitions.
            config (dict): The configuration dictionary.
            seed (int): The random seed for reproducibility.
            steps_done (int): The number of steps done in the environment.
        """
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
        """
        Selects an action based on the current state using an epsilon-greedy policy.

        Args:
            state (torch.Tensor): The current state of the environment.

        Returns:
            torch.Tensor: The selected action as a tensor.
        """
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
        """
        Optimize the model by sampling a batch of transitions from memory, computing the loss, and performing a gradient descent step.

        This method performs the following steps:
        1. Checks if there are enough transitions in memory to sample a batch.
        2. Samples a batch of transitions from memory.
        3. Separates the batch into states, actions, rewards, and next states.
        4. Computes the state-action values for the current states using the policy network.
        5. Computes the next state values using the target network.
        6. Computes the expected state-action values using the rewards and next state values.
        7. Computes the loss between the state-action values and the expected state-action values.
        8. Performs a backward pass to compute the gradients.
        9. Clamps the gradients to avoid exploding gradients.
        10. Updates the policy network parameters using the optimizer.
        11. Resets the target and policy networks.

        Returns:
            float: The loss value.
        """
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
        """
        Train the Deep Q-Network (DQN) model.

        Args:
            num_episodes (int): Number of episodes to train the model.
            model_dir (str): Directory to save the trained model.
            log_dir (str): Directory to save the training logs.
            log_name (str): Name for the log entries.
            model_name (str): Name for the saved model files.
        """
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
