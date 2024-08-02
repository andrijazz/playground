from __future__ import absolute_import, division

from abc import ABCMeta, abstractmethod


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def add_to_replay_buffer(self, paths):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass
