from alfworld_grpo.tools.env import get_env

def go_to(receptacle: str)->str:
    """Move to a receptacle.

    Args:
        receptacle: A piece of furniture that may contain objects on it

    Returns:
        The observation after moving to the receptacle.

    Examples:
        "countertop 1" -> "You arrive at countertop 1. On it you see ..."
    """
    env = get_env()
    action = f"go to {receptacle}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]