from alfworld_grpo.tools.env import get_env

def reset_env()->str:
    """Resets the Alfworld text environment.

    Returns:
        The initial observation with the task specification after reseting the environment.

    Examples:
        "" -> "Welcome to TextWorld, ALFRED! ... Your task is to: ..."
    """
    env = get_env()
    obs, info = env.reset()
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]