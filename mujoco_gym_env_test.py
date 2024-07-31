from itertools import count
import gymnasium as gym


env = gym.make('Ant-v4', render_mode='human')
observation, info = env.reset()

episode_reward = 0
for t in count():
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_reward += reward

    if done:
        print(f"Total reward of the episode: {episode_reward:7.2f}")
        break

env.close()
