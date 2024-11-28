# 融合脉冲神经网络与深度强化学习

系统与计算神经科学课程作业

本项目探讨脉冲神经网络与深度强化学习结合的潜力，通过解决经典强化学习问题——CartPole，来验证这种结合的表现和优势。

## 实验环境

- **操作系统**：Windows 11
- **编程语言**：Python 3.10
- **编程工具**：Visual Studio Code
- **主要库及版本**：
  - Pytorch 2.5.1
  - Gymnasium 1.0.0
  - SpikingJelly 0.0.0.0.14

## 项目结构

- ``Exp``： 实验代码文件夹
  - `algorithm`：使用算法文件夹
    - `base`：基础算法
      - `Neuron.py`：非脉冲神经网络神经元实现
    - `DQN.py`：深度Q脉冲网络与记忆回放机制实现
  - `data`：数据文件夹
    - `x-y`：x个隐藏层，y个神经元的数据文件夹
      - `x-y-DQSN-Loss.csv`：训练过程中的损失数据
      - `x-y-DQSN-Reward.csv`：训练过程中的奖励数据
      - `Trail_z.csv`：第z次测试数据
    - `DQN`：深度Q脉冲网络数据文件夹
      - ``DQSN-Loss.csv``：训练过程中的损失数据
      - ``DQSN-Reward.csv``：训练过程中的奖励数据
  - `img`：图片文件夹
  - `log`：训练过程日志文件夹
  - `model`：模型文件夹
    - `Layer-x-Hidden-y-T-z-DQSN-CPU-Fixed-best_model.pt`：x层隐藏层，y个神经元，z个时间步长 训练过程中最佳DQSN模型
    - `Layer-x-Hidden-y-T-z-DQSN-CPU-Fixed-final_model.pt`：x层隐藏层，y个神经元，z个时间步长 训练结束DQSN模型
  - `plot.ipynb`：绘图文件
  - `DQNTrainer`：深度Q脉冲网络训练器
  - `DQNPlayer`：深度Q脉冲网络测试器
  - `main.py`：主程序
  - `requirements.txt`：依赖库文件

- ``Rep``：实验报告文件夹
  - ``img``：图片文件夹
  - `Report.pdf`：实验报告PDF版本
  - `Report.typ`：实验报告源文件
