from benchmarl.algorithms.gire_mappo import GireMappoConfig
from benchmarl.tasks import CamarTask
from benchmarl.environments import CamarTask
from benchmarl.experiment import Experiment

# Just test env reset directly
from benchmarl.environments.camar.common import CamarTask
env = CamarTask.RANDOM_GRID.get_from_yaml()
env_instance = env.get_env_ops(1, None)
td = env_instance.reset()
print("Keys in reset td:", td.keys(True, True))
