from projects.imitation.base_agent import BaseAgent
from projects.imitation.replay_buffer import ReplayBuffer
from projects.imitation.mlp_policy import MLPPolicy


class BCAgent(BaseAgent):
    """
    Behavioral cloning agent
    """
    def __init__(self, **kwargs):
        super(BCAgent, self).__init__(**kwargs)
        self.actor = MLPPolicy()
        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        self.actor.update(ob_no, ac_na)

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
