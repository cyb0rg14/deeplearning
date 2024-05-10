from pettingzoo.classic import tictactoe_v3

env = tictactoe_v3.env(render_mode='human')
env.reset(seed=42)

actions = env.action_space
observation = env.observation_space
print(actions, observation)


# for agent in env.agent_iter():
#     observation, reward, terminated, truncated, _ = env.last()

#     if terminated or truncated:
#         action = None 
#     else:
#         mask = observation['action_mask']
#         action = env.action_space(agent).sample(mask)

#     env.step(action)
# env.close()