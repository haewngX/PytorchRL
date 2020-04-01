import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import random
import time
from agent.sac1 import SAC
from hyperparameters import Config

config = Config()

config.name = 'Pendulum-v0'


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action


# 初始化环境
env = NormalizedActions(gym.make(config.name))

# 初始化agent
agent = SAC(config=config, env=env)

# agent.load()

iteration = 0  # 总步数

# 记录时间
start_time = time.time()
ep_reward_list = []  # 存放每回合的reward
mean_ep_reward_list = []  # 整个训练过程的平均reward

# 循环nepisode个episode
for e in range(config.episode):
    s = env.reset()  # 初始化state
    ep_reward = 0  # 初始化每回合的reward
    # 每个回合循环episode_step步
    # for t in range(episode_step):
    while True:
        env.render()  # 显示图形
        a = agent.choose_action(s)  # 选择动作
        s_, r, done, _ = env.step(a)  # 与环境交互得到下一个状态 奖励和done
        agent.store_transition(s, a, r, s_, done)  # 储存记忆
        s = s_  # 更新状态
        ep_reward += r  # 更新当前回合reward
        # learn
        if iteration >= agent.learn_start:
            agent.learn()
        # 到达终止状态，显示信息，跳出循环
        if done:
            # 计算运行时间
            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            # 输出该回合的累计回报等信息
            print('Ep: %d ep_reward: %.2f iteration: %d time: %d:%02d:%02d' % (e, ep_reward,
                                                                                           iteration,
                                                                                           h, m, s))
            ep_reward_list.append(ep_reward)
            average = np.mean(np.array(ep_reward_list))
            mean_ep_reward_list.append(average)
            # agent.save()
            break
        iteration += 1
# 画图
plt.plot(range(len(ep_reward_list)), ep_reward_list, color="red", label="ep_reward", linewidth=1.5, linestyle='--')
plt.plot(range(len(mean_ep_reward_list)), mean_ep_reward_list, color="green", label="mean_reward", linewidth=1.5)
size = 12
plt.xticks(fontsize=size)  # 默认字体大小为10
plt.yticks(fontsize=size)
plt.ylabel('reward')
plt.xlabel('episodes')
plt.title('Pendulum-v0', fontsize=size)
plt.legend()
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=size)  # 设置图例字体的大小和粗细
plt.savefig('sac1 reward.png')
plt.show()