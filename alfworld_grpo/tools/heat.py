from alfworld_grpo.tools.env import get_env

def heat(object: str, receptacle: str)->str:
    """Heat an object using a receptacle.

    Args:
        object: A manipulable household object.
        receptacle: A piece of furniture that can be used to heat objects.

    Returns:
        The observation after heating the object using the receptacle.

    Examples:
        "apple 1, microwave 1" -> "You heat the apple 1 using microwave 1"
    """
    env = get_env()
    action = f"heat {object} with {receptacle}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]