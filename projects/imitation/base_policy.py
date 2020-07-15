from __future__ import absolute_import, division

from abc import ABCMeta, abstractmethod


class BasePolicy(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(BasePolicy, self).__init__(**kwargs)

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def get_action(self, obs):
        pass

    @abstractmethod
    def update(self, obs, acs):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def restore(self, filepath):
        pass
