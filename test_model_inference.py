#!/usr/bin/env python3
"""
Simple model inference test for the trained CNN+LSTM stress detection model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def test_model_inference():
    """Test the trained model with synthetic data"""
    
    print("🔍 Testing Trained CNN+LSTM Stress Detection Model")
    print("=" * 60)
    
    # Check if model exists
    model_path = 'stress_detection_cnn_lstm.h5'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Load the trained model
        print("📁 Loading trained model...")
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Print model summary
        print("\n🏗️ Model Architecture:")
        model.summary()
        
        # Create synthetic test data (same format as training)
        print("\n🧪 Creating synthetic test data...")
        n_samples = 5
        sequence_length = 240
        n_features = 14
        
        # Generate random physiological signals
        test_data = np.random.randn(n_samples, sequence_length, n_features)
        
        # Add some realistic patterns
        for i in range(n_samples):
            # Simulate different stress patterns
            if i % 2 == 0:  # Even samples - simulate stress
                test_data[i, :, 0] += np.sin(np.linspace(0, 4*np.pi, sequence_length)) * 0.5  # EDA
                test_data[i, :, 1] += 80 + np.random.normal(0, 10, sequence_length)  # HR
            else:  # Odd samples - simulate relaxed state
                test_data[i, :, 0] += np.sin(np.linspace(0, 2*np.pi, sequence_length)) * 0.2  # EDA
                test_data[i, :, 1] += 65 + np.random.normal(0, 5, sequence_length)  # HR
        
        print(f"📊 Test data shape: {test_data.shape}")
        
        # Make predictions
        print("\n🔮 Making predictions...")
        predictions = model.predict(test_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        # Display results
        print("\n📋 Prediction Results:")
        print("-" * 50)
        class_names = ['Non-Stress', 'Stress']
        
        for i in range(n_samples):
            predicted_class = class_names[predicted_classes[i]]
            confidence = confidence_scores[i] * 100
            stress_prob = predictions[i][1] * 100
            non_stress_prob = predictions[i][0] * 100
            
            print(f"\nSample {i+1}:")
            print(f"  Predicted: {predicted_class}")
            print(f"  Confidence: {confidence:.1f}%")
            print(f"  Stress probability: {stress_prob:.1f}%")
            print(f"  Non-stress probability: {non_stress_prob:.1f}%")
            
            # Add emoji indicators
            if predicted_classes[i] == 1:  # Stress
                print(f"  Status: 😰 STRESS DETECTED")
            else:  # Non-stress
                print(f"  Status: 😌 RELAXED STATE")
        
        # Test with extreme cases
        print("\n🎯 Testing with extreme cases...")
        
        # High stress pattern
        stress_sample = np.random.randn(1, sequence_length, n_features)
        stress_sample[0, :, 0] += 2.0  # High EDA
        stress_sample[0, :, 1] += 100  # High heart rate
        
        stress_pred = model.predict(stress_sample, verbose=0)
        stress_class = np.argmax(stress_pred)
        
        print(f"\nHigh Stress Pattern:")
        print(f"  Prediction: {class_names[stress_class]}")
        print(f"  Stress probability: {stress_pred[0][1]*100:.1f}%")
        
        # Low stress pattern
        relax_sample = np.random.randn(1, sequence_length, n_features)
        relax_sample[0, :, 0] -= 0.5  # Low EDA
        relax_sample[0, :, 1] += 60   # Low heart rate
        
        relax_pred = model.predict(relax_sample, verbose=0)
        relax_class = np.argmax(relax_pred)
        
        print(f"\nRelaxed Pattern:")
        print(f"  Prediction: {class_names[relax_class]}")
        print(f"  Stress probability: {relax_pred[0][1]*100:.1f}%")
        
        print("\n✨ Model inference test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during model testing: {str(e)}")
        return

if __name__ == "__main__":
    test_model_inference()
