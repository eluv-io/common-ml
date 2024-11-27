from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

@dataclass
class Data(ABC):
    """
    This class is useful for converting data to and from a dictionary or url request. Often explicit type conversion and verification is 
    necessary and this class provides a common interface for that.

    The `from_dict` method is a class method that should be implemented by subclasses, it should take a dictionary and return an instance of the class 
    while raising an exception if the dictionary is not valid.
    """
    @staticmethod
    @abstractmethod
    def from_dict(data: dict) -> 'Data':
        pass

    def to_dict(self) -> dict:
        return asdict(self)