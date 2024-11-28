import os
from itertools import count

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from algorithm.DQN import DQSN
from spikingjelly.activation_based import functional, monitor, neuron


class DQNPlayer:
    """
    A class to represent a Deep Q-Network (DQN) player.
    Attributes
    ----------
    device : torch.device
        The device to run the computations on (CPU or GPU).
    env : gym.Env
        The environment in which the agent will play.
    n_states : int
        The number of states in the environment.
    n_actions : int
        The number of possible actions in the environment.
    policy_net : DQSN
        The policy network used by the agent.
    T : int
        The time constant for the spiking neural network.
    actions : list
        A list to store the actions taken by the agent.
    rewards : list
        A list to store the rewards received by the agent.
    Methods
    -------
    play(played_frames=60, save_fig_num=0, fig_dir=None, figsize=(12, 6), firing_rates_plot_type="bar", heatmap_shape=None):
        Plays and visualizes the game process of the DQN.
    _plot_voltage(LIF_v, action, delta_lim, plot_type):
        Plots the voltage of LIF neurons at the last time step.
    _plot_firing_rates(firing_rates, plot_type, heatmap_shape):
        Plots the firing rates of IF neurons.
    _update_env(state, action, i, over_score):
        Interacts with the environment and updates the state.
    _plot_game_screen(state, i, save_fig_num, fig_dir, plot_type):
        Renders and plots the game screen.
    """

    def __init__(self, use_cuda, env_name, hidden_size, pt_path, T=16):
        """
        Initializes the DQNPlayer.
        Args:
            use_cuda (bool): If True, use CUDA for computation.
            env_name (str): The name of the environment to create.
            hidden_size (int): The size of the hidden layer in the neural network.
            pt_path (str): The path to the pre-trained model.
            T (int, optional): The time step parameter for the DQSN model. Default is 16.
        Attributes:
            device (torch.device): The device to run the computations on.
            env (gym.Env): The environment instance.
            n_states (int): The number of states in the environment.
            n_actions (int): The number of actions in the environment.
            policy_net (DQSN): The policy network model.
            T (int): The time step parameter for the DQSN model.
            actions (list): List to store actions taken.
            rewards (list): List to store rewards received.
        """
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.env = gym.make(env_name, render_mode="rgb_array_list").unwrapped
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.policy_net = DQSN(self.n_states, hidden_size, self.n_actions, T).to(
            self.device
        )
        self.policy_net.load_state_dict(torch.load(pt_path, map_location=self.device))
        self.T = T

        self.actions = []
        self.rewards = []

    def play(
        self,
        played_frames=60,
        save_fig_num=0,
        fig_dir=None,
        figsize=(12, 6),
        firing_rates_plot_type="bar",
        heatmap_shape=None,
    ):
        """
        Simulates playing the game using the trained policy network.
        Args:
            played_frames (int, optional): Number of frames to play. Defaults to 60.
            save_fig_num (int, optional): Number of figures to save. Defaults to 0.
            fig_dir (str, optional): Directory to save figures. Defaults to None.
            figsize (tuple, optional): Size of the figure for plotting. Defaults to (12, 6).
            firing_rates_plot_type (str, optional): Type of plot for firing rates ("bar" or other types). Defaults to "bar".
            heatmap_shape (tuple, optional): Shape of the heatmap for firing rates. Defaults to None.
        Returns:
            None
        """
        plt.rcParams["figure.figsize"] = figsize
        plt.ion()

        self.env.reset()
        state = torch.zeros([1, self.n_states], dtype=torch.float, device=self.device)
        spike_seq_monitor = monitor.OutputMonitor(self.policy_net, neuron.IFNode)

        delta_lim = 0
        over_score = 1e9

        with torch.no_grad():
            for i in count():
                # Get the action from the policy network
                LIF_v = self.policy_net(state)
                action = LIF_v.max(1)[1].view(1, 1).item()

                self.actions.append(action)

                # Visualize the voltage of LIF neurons at the last time step
                # self._plot_voltage(LIF_v, action, delta_lim, firing_rates_plot_type)

                # Get the firing rates of IF neurons
                IF_spikes = torch.cat(spike_seq_monitor.records, 0)
                firing_rates = IF_spikes.mean(axis=0)
                # self._plot_firing_rates(
                #     firing_rates, firing_rates_plot_type, heatmap_shape
                # )

                # Reset the network
                functional.reset_net(self.policy_net)

                # Interact with the environment and update the state
                reward, subtitle, done, state = self._update_env(
                    state, action, i, over_score
                )

                # Store the reward
                self.rewards.append(reward)

                # plt.suptitle(subtitle)

                # Plot the game screen
                # self._plot_game_screen(
                #     state, i, save_fig_num, fig_dir, firing_rates_plot_type
                # )

                # if done and i >= played_frames:
                if done:
                    self.env.close()
                    plt.close()
                    break

    def _plot_voltage(self, LIF_v, action, delta_lim, plot_type):
        if plot_type == "bar":
            plt.subplot2grid((2, 9), (1, 0), colspan=3)
        elif plot_type == "heatmap":
            plt.subplot2grid((2, 3), (1, 0))

        plt.xticks(np.arange(2), ("Left", "Right"))
        plt.ylabel("Voltage")
        plt.title("Voltage of LIF neurons at last time step")
        delta_lim = (LIF_v.max() - LIF_v.min()) * 0.5
        plt.ylim(LIF_v.min() - delta_lim, LIF_v.max() + delta_lim)
        plt.yticks([])
        plt.text(0, LIF_v[0][0], str(round(LIF_v[0][0].item(), 2)), ha="center")
        plt.text(1, LIF_v[0][1], str(round(LIF_v[0][1].item(), 2)), ha="center")

        plt.bar(
            np.arange(2),
            LIF_v.squeeze(),
            color=["r", "gray"] if action == 0 else ["gray", "r"],
            width=0.5,
        )

        if LIF_v.min() - delta_lim < 0:
            plt.axhline(0, color="black", linewidth=0.1)

    def _plot_firing_rates(self, firing_rates, plot_type, heatmap_shape):
        if plot_type == "bar":
            plt.subplot2grid((2, 9), (0, 4), rowspan=2, colspan=5)
        elif plot_type == "heatmap":
            plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)

        plt.title("Firing rates of IF neurons")

        if plot_type == "bar":
            plt.xlabel("Neuron index")
            plt.ylabel("Firing rate")
            plt.xlim(0, firing_rates.size(0))
            plt.ylim(0, 1.01)
            plt.bar(np.arange(firing_rates.size(0)), firing_rates, width=0.5)

        elif plot_type == "heatmap":
            heatmap = plt.imshow(
                firing_rates.reshape(heatmap_shape), vmin=0, vmax=1, cmap="ocean"
            )
            plt.gca().invert_yaxis()
            cbar = heatmap.figure.colorbar(heatmap)
            cbar.ax.set_ylabel("Magnitude", rotation=90, va="top")

    def _update_env(self, state, action, i, over_score):
        subtitle = f"Position={state[0][0].item(): .2f}, Velocity={state[0][1].item(): .2f}, Pole Angle={state[0][2].item(): .2f}, Pole Velocity At Tip={state[0][3].item(): .2f}, Score={i}"
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if done:
            over_score = min(over_score, i)
            subtitle = f"Game over, Score={over_score}"

        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        return reward, subtitle, done, obs

    def _plot_game_screen(self, state, i, save_fig_num, fig_dir, plot_type):
        screen = self.env.render().copy()
        screen[300, :, :] = 0

        if plot_type == "bar":
            plt.subplot2grid((2, 9), (0, 0), colspan=3)
        elif plot_type == "heatmap":
            plt.subplot2grid((2, 3), (0, 0))

        plt.xticks([])
        plt.yticks([])
        plt.title("Game screen")
        plt.imshow(screen, interpolation="bicubic")
        plt.pause(0.001)

        if i < save_fig_num:
            plt.savefig(os.path.join(fig_dir, f"{i}.png"))
