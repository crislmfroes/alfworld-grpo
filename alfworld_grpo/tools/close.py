from alfworld_grpo.tools.env import get_env

def close(receptacle: str)->str:
    """Close a receptacle.

    Args:
        receptacle: A piece of furniture that may contain objects on it.

    Returns:
        The observation after closing the receptacle.

    Examples:
        "drawer 1" -> "You close the drawer 1"
    """
    env = get_env()
    action = f"close {receptacle}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]