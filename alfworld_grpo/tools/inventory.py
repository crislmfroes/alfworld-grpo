from alfworld_grpo.tools.env import get_env

def inventory()->str:
    """Check your current inventory.

    Returns:
        The observation after checking your inventory.

    Examples:
        "" -> "On your inventory you have ..."
    """
    env = get_env()
    action = f"inventory"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]