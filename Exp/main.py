import pandas as pd

from algorithm.DQN import DQSN
from DQNPlayer import DQNPlayer
from DQNTrainer import DQNTrainer

if __name__ == "__main__":
    CONFIG = {
        "BATCH_SIZE": 128,
        "GAMMA": 0.999,
        "EPS_START": 0.9,
        "EPS_END": 0.05,
        "EPS_DECAY": 200,
        "TARGET_UPDATE": 10,
        "MEMORY_CAPACITY": 10000,
    }

    # trainer = DQNTrainer(
    #     "CartPole-v0", DQSN, hidden_size=512, t=8, use_cuda=False, config=CONFIG
    # )
    # trainer.train(
    #     num_episodes=500,
    #     model_dir="./model",
    #     log_dir="./log",
    #     model_name="Layer-1-Hidden-512-T-8-DQSN-CPU-Fixed",
    #     log_name="Layer-1-Hidden-512-T-8-DQSN-CPU-Fixed",
    # )

    for i in range(77, 101):
        player = DQNPlayer(
            use_cuda=False,
            env_name="CartPole-v1",
            hidden_size=512,
            pt_path=f"model\Layer-1-Hidden-512-T-16-DQSN-CPU-Fixed_best_model.pt",
            T=16,
        )
        player.play(played_frames=100)

        # 打印或保存 actions 和 rewards
        # print("Actions:", player.actions)
        # print("Rewards:", player.rewards)
        print(f"Trail_{i}_Total Reward:", sum(player.rewards))
        # print("Action Selection",  pd.value_counts(player.actions))
        df = pd.DataFrame({"Actions": player.actions, "Rewards": player.rewards})
        df.to_csv(f"data/16-512/Trail_{i}.csv")
