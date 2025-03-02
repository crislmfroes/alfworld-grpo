from alfworld_grpo.tools.env import get_env

def look()->str:
    """Look around your current location.

    Returns:
        The observation after looking around.

    Examples:
        "" -> "Looking around you see ..."
    """
    env = get_env()
    action = f"look"
    obs, score, done, info = env.step([action])
    if info['won'][0] == True:
        return "SUCCESS! Your task is now complete!"
    return obs[0]