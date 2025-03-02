from alfworld_grpo.tools.env import get_env

def move_to(object: str, receptacle: str)->str:
    """Place an object in or on a receptacle.

    Args:
        object: A manipulable household object.
        receptacle: A piece of furniture that may contain objects on it.

    Returns:
        The observation after placing the object on the receptacle.

    Examples:
        "apple 1, drawer 1" -> "You move the apple 1 to drawer 1"
    """
    env = get_env()
    action = f"move {object} to {receptacle}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]