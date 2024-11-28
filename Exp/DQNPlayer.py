import os
from itertools import count

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from spikingjelly.activation_based import functional, monitor, neuron

from algorithm.DQN import DQSN


class DQNPlayer:
    def __init__(self, use_cuda, env_name, hidden_size, pt_path, T=16):
        """
        初始化 DQN Player
        :param use_cuda: 是否使用 GPU
        :param env_name: 环境名称
        :param hidden_size: 隐藏层大小
        :param pt_path: 模型权重文件路径
        :param T: 时间步数，用于 SNN 模型
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

        # 用于保存 action 和 reward
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
        播放并可视化 DQN 的游戏过程
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
                # 获取动作
                LIF_v = self.policy_net(state)  # shape=[1, 2]
                action = LIF_v.max(1)[1].view(1, 1).item()

                # 保存 action
                self.actions.append(action)

                # 可视化神经元电压
                # self._plot_voltage(LIF_v, action, delta_lim, firing_rates_plot_type)

                # 获取并绘制神经元的放电频率
                IF_spikes = torch.cat(spike_seq_monitor.records, 0)
                firing_rates = IF_spikes.mean(axis=0)
                # self._plot_firing_rates(
                #     firing_rates, firing_rates_plot_type, heatmap_shape
                # )

                # 重置网络状态
                functional.reset_net(self.policy_net)

                # 环境交互并更新状态
                reward, subtitle, done, state = self._update_env(
                    state, action, i, over_score
                )

                # 保存 reward
                self.rewards.append(reward)

                # plt.suptitle(subtitle)

                # 渲染游戏画面
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
        screen[300, :, :] = 0  # 画出黑线

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
