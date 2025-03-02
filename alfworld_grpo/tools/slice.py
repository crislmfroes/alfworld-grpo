from alfworld_grpo.tools.env import get_env

def slice(object: str, sharp_object: str)->str:
    """Slice an object using a sharp object.

    Args:
        object: A manipulable household object.
        sharp_object: A sharp manipulable household object.

    Returns:
        The observation after slicing the object using the sharp object.

    Examples:
        "apple 1, knife 1" -> "You slice the apple 1 using knife 1"
    """
    env = get_env()
    action = f"slice {object} with {sharp_object}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]