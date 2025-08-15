#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class StressDetectionInference:
    
    def __init__(self, model_path='stress_detection_cnn_lstm.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = ['Non-Stress', 'Stress']
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = keras.models.load_model(self.model_path)
        return True
    
    def predict(self, signal_data, return_probabilities=True):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if len(signal_data.shape) == 2:
            signal_data = np.expand_dims(signal_data, axis=0)
        
        predictions = self.model.predict(signal_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        results = []
        for i in range(len(predictions)):
            result = {
                'predicted_class': self.class_names[predicted_classes[i]],
                'confidence': float(confidence_scores[i]),
                'stress_probability': float(predictions[i][1]),
                'non_stress_probability': float(predictions[i][0])
            }
            
            if return_probabilities:
                result['raw_probabilities'] = predictions[i].tolist()
            
            results.append(result)
        
        return results if len(results) > 1 else results[0]
    
    def generate_test_data(self, n_samples=5, sequence_length=240, n_features=14):
        test_data = np.random.randn(n_samples, sequence_length, n_features)
        
        for i in range(n_samples):
            if i % 2 == 0:
                test_data[i, :, 0] += np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 0.5
                test_data[i, :, 1] += 80 + np.random.normal(0, 10, sequence_length)
            else:
                test_data[i, :, 0] += np.sin(np.linspace(0, 2*np.pi, sequence_length)) * 0.2
                test_data[i, :, 1] += 65 + np.random.normal(0, 5, sequence_length)
        
        return test_data
    
    def run_performance_test(self):
        if self.model is None:
            self.load_model()
        
        test_data = self.generate_test_data()
        results = self.predict(test_data)
        
        performance_metrics = {
            'total_samples': len(results) if isinstance(results, list) else 1,
            'average_confidence': np.mean([r['confidence'] for r in (results if isinstance(results, list) else [results])]),
            'stress_detections': sum(1 for r in (results if isinstance(results, list) else [results]) if r['predicted_class'] == 'Stress'),
            'model_loaded': True,
            'inference_successful': True
        }
        
        return performance_metrics, results

def main():
    detector = StressDetectionInference()
    
    try:
        detector.load_model()
        performance, results = detector.run_performance_test()
        
        print("Stress Detection Model - Performance Test")
        print("=" * 50)
        print(f"Model Status: Loaded Successfully")
        print(f"Total Samples: {performance['total_samples']}")
        print(f"Average Confidence: {performance['average_confidence']:.2%}")
        print(f"Stress Detections: {performance['stress_detections']}")
        print()
        
        if isinstance(results, list):
            for i, result in enumerate(results):
                print(f"Sample {i+1}:")
                print(f"  Prediction: {result['predicted_class']}")
                print(f"  Confidence: {result['confidence']:.2%}")
                print(f"  Stress Probability: {result['stress_probability']:.2%}")
                print()
        else:
            print("Single Sample Result:")
            print(f"  Prediction: {results['predicted_class']}")
            print(f"  Confidence: {results['confidence']:.2%}")
            print(f"  Stress Probability: {results['stress_probability']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
