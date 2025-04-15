from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
import time
from pathlib import Path

# Abstract base class - all models need to inherit from this class
class BaseModel(ABC):
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Model prediction interface"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name"""
        pass