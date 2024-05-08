import gymnasium as gym

env = gym.make('LunarLander-v2', render_mode='human')
observation, info = env.reset()

# print(env.observation_space)
# print(env.action_space.n)

for _ in range(1000):
    action = env.action_space.sample()
    new_state, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()