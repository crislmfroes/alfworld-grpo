from alfworld_grpo.tools.env import get_env

def use(object: str)->str:
    """Use an object.

    Args:
        object: A manipulable household object.

    Returns:
        The observation after using the object.

    Examples:
        "floorlamp 1" -> "You toggle on the floorlamp 1"
    """
    env = get_env()
    action = f"use {object}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]