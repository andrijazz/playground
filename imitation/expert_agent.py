from projects.imitation.base_agent import BaseAgent
from projects.imitation.loaded_gaussian_policy import LoadedGaussianPolicy


class ExpertAgent(BaseAgent):
    def __init__(self, filepath, **kwargs):
        super(ExpertAgent, self).__init__(**kwargs)
        self.actor = LoadedGaussianPolicy(filepath)

    def train(self):
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError
