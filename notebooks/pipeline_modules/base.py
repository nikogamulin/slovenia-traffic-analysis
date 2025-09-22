
"""Base pipeline components."""
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class PipelineResult:
    """Result container for pipeline operations."""
    success: bool
    data: Optional[pd.DataFrame] = None
    message: str = ''
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class PipelineComponent(ABC):
    """Abstract base class for pipeline components."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup component logger."""
        logger = logging.getLogger(f'pipeline.{self.name}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> PipelineResult:
        """Process data through this component."""
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data."""
        if data is None or data.empty:
            self.logger.error(f'Empty or None data received in {self.name}')
            return False
        return True
