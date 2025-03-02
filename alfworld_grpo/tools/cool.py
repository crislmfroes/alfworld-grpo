from alfworld_grpo.tools.env import get_env

def cool(object: str, receptacle: str)->str:
    """Cool an object using a receptacle.

    Args:
        object: A manipulable household object.
        receptacle: A piece of furniture that can be used to cool objects.

    Returns:
        The observation after cooling the object using the receptacle.

    Examples:
        "apple 1, fridge 1" -> "You cool the apple 1 using fridge 1"
    """
    env = get_env()
    action = f"cool {object} with {receptacle}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]