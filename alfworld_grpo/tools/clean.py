from alfworld_grpo.tools.env import get_env

def clean(object: str, receptacle: str)->str:
    """Clean an object using a receptacle.

    Args:
        object: A manipulable household object.
        receptacle: A piece of furniture that can be used to clean objects.

    Returns:
        The observation after cleaning the object using the receptacle.

    Examples:
        "fork 1, sinkbasin 1" -> "You clean the fork 1 using sinkbasin 1"
    """
    env = get_env()
    action = f"clean {object} with {receptacle}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]