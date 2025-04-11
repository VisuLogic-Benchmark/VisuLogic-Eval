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

# Data loader base class
class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_data(self):
        """Implement specific data loading logic"""
        raise NotImplementedError
    
    def preprocess(self, data):
        """Data preprocessing"""
        return data

# Evaluation metric base class
class Metric:
    def __init__(self, name: str):
        self.name = name
    
    def calculate(self, predictions: List, ground_truth: List) -> float:
        """Calculate evaluation metric"""
        raise NotImplementedError

# Benchmark framework
class Benchmark:
    def __init__(self, 
                 data_loader: DataLoader,
                 metrics: List[Metric]):
        self.data_loader = data_loader
        self.metrics = metrics
        self.results = {}
        
    def evaluate_model(self, model: BaseModel) -> Dict:
        """Evaluate a single model"""
        # Load data
        data = self.data_loader.load_data()
        predictions = []
        ground_truth = []
        
        # Record inference time
        start_time = time.time()
        
        # Get model prediction results
        for item in data:
            pred = model.predict(item)
            predictions.append(pred)
            ground_truth.append(item['label'])  # Assume the data contains a label field
            
        inference_time = time.time() - start_time
        
        # Calculate all evaluation metrics
        metric_results = {}
        for metric in self.metrics:
            score = metric.calculate(predictions, ground_truth)
            metric_results[metric.name] = score
            
        # Record results
        result = {
            'model_name': model.name,
            'metrics': metric_results,
            'inference_time': inference_time
        }
        
        self.results[model.name] = result
        return result
    
    def run_benchmark(self, models: List[BaseModel]) -> Dict:
        """Run benchmark tests for all models"""
        for model in models:
            self.evaluate_model(model)
        return self.results
    
    def save_results(self, output_path: str):
        """Save benchmark results"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

# Example implementation
class AccuracyMetric(Metric):
    def __init__(self):
        super().__init__('accuracy')
    
    def calculate(self, predictions: List, ground_truth: List) -> float:
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        return correct / len(predictions)

# Example usage
class MyModelA(BaseModel):
    @property
    def name(self):
        return "ModelA"
        
    def predict(self, input_data):
        # Implement specific prediction logic
        pass


def main():
    # Create data loader
    data_loader = DataLoader("path/to/data")
    
    # Create evaluation metrics
    metrics = [AccuracyMetric()]
    
    # Create benchmark instance
    benchmark = Benchmark(data_loader, metrics)
    
    # Create a list of models to test
    models = [MyModelA()]
    
    # Run benchmark
    results = benchmark.run_benchmark(models)
    
    # Save results
    benchmark.save_results("benchmark_results.json")

if __name__ == "__main__":
    main()