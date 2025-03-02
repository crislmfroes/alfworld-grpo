from alfworld_grpo.tools.env import get_env

env = get_env()
obs, info = env.reset()
print(obs[0])
obs, score, done, info = env.step(['help'])
print(obs[0])