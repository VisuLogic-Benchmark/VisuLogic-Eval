class Metric:
    def __init__(self, name: str):
        self.name = name
    
    def calculate(self, predictions: List, ground_truth: List) -> float:
        """计算评估指标"""
        raise NotImplementedError
