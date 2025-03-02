from alfworld_grpo.tools.env import get_env

def examine(something: str)->str:
    """Examine a receptacle or an object.

    Args:
        something: A manipulable household object, or a piece of furniture that may contain objects on it.

    Returns:
        The observation with details about the object or receptacle.

    Examples:
        "apple 1" -> "This is a normal apple 1"
    """
    env = get_env()
    action = f"examine {something}"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]