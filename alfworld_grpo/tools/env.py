from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

try:
    if env == None:
        # load config
        config = generic.load_config()
        env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

        # setup environment
        env = get_environment(env_type)(config, train_eval='train')
        env = env.init_env(batch_size=1)
except:
    # load config
    config = generic.load_config()
    env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

    # setup environment
    env = get_environment(env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)

def get_env():
    global env
    return env