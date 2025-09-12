from abc import ABC, abstractmethod

class map_base(ABC):

    @abstractmethod
    def map(self, feat, bs):
        pass

    @abstractmethod
    def get_iter(self, i):
        pass