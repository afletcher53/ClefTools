# abstract class for API classes

from abc import ABC, abstractmethod


class API(ABC):
    @abstractmethod
    def __init__(self):
        # base url for the API
        self.base_url = None

        pass

    @abstractmethod
    def get(self, *args, **kwargs):
        pass
