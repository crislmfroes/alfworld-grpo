from alfworld_grpo.tools.env import get_env

def open(receptacle: str)->str:
    """Open a receptacle.

    Args:
        receptacle: A piece of furniture that may contain objects on it.

    Returns:
        The observation after opening the receptacle.

    Examples:
        "drawer 1" -> "You open drawer 1. On it you see ..."
    """
    env = get_env()
    action = f"open {receptacle}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]