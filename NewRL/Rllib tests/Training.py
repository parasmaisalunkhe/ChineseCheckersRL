from ray.tune.registry import register_env

def env_creator(env_config):
    return CustomMAEnv(env_config)

register_env("CustomMAEnv-v0", env_creator)
