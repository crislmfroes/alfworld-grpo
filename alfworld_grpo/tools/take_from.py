from alfworld_grpo.tools.env import get_env

def take_from(object: str, receptacle: str)->str:
    """Take an object from a receptacle.

    Args:
        object: A manipulable household object.
        receptacle: A piece of furniture that may contain objects on it.

    Returns:
        The observation after taking the object from the receptacle.

    Examples:
        "apple 1, drawer 1" -> "You take the apple 1 from drawer 1"
    """
    env = get_env()
    action = f"take {object} from {receptacle}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]