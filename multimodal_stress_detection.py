import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from deepface import DeepFace
import cv2
from PIL import Image

# Configure GPU for inference
def configure_gpu_for_inference():
    """Configure GPU settings for inference"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured for inference: {gpus[0].name}")
            return True
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("⚠️  No GPU found. Using CPU for inference.")
        return False

# Initialize GPU
gpu_available = configure_gpu_for_inference()

class MultimodalStressDetector:
    """
    Advanced stress detection system combining:
    1. Physiological signals (CNN+LSTM model)
    2. Facial emotion analysis (DeepFace)
    """
    
    def __init__(self, physiological_model_path='stress_detection_cnn_lstm.h5'):
        """Initialize the multimodal stress detector"""
        try:
            self.physio_model = load_model(physiological_model_path)
            print(f"✅ Loaded physiological model from {physiological_model_path}")
        except Exception as e:
            print(f"❌ Could not load physiological model: {e}")
            self.physio_model = None
        
        print("🔄 Initializing DeepFace for facial emotion detection...")
        try:
            # Test DeepFace initialization
            test_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            DeepFace.analyze(test_img, actions=['emotion'], enforce_detection=False)
            print("✅ DeepFace initialized successfully")
        except Exception as e:
            print(f"⚠️ DeepFace initialization warning: {e}")
    
    def predict_physiological_stress(self, signal_sequence):
        """
        Predict stress from physiological signals
        Returns DeepFace-style output format
        """
        if self.physio_model is None:
            return {"error": "Physiological model not loaded"}
        
        try:
            if len(signal_sequence.shape) == 2:
                signal_sequence = signal_sequence.reshape(1, signal_sequence.shape[0], signal_sequence.shape[1])
            
            # Force prediction on GPU if available
            with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
                prediction = self.physio_model.predict(signal_sequence, verbose=0)[0]
            
            result = {
                'stress': round(float(prediction[1]) * 100, 2),
                'non_stress': round(float(prediction[0]) * 100, 2),
                'dominant_state': 'stress' if prediction[1] > prediction[0] else 'non_stress',
                'confidence': round(float(max(prediction)) * 100, 2),
                'source': 'physiological_signals',
                'device': '/GPU:0' if gpu_available else '/CPU:0'
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Physiological prediction failed: {e}"}
    
    def analyze_facial_emotion(self, image_path_or_array):
        """
        Analyze facial emotions using DeepFace
        Returns stress-related interpretation
        """
        try:
            # Analyze emotions
            result = DeepFace.analyze(
                img_path=image_path_or_array,
                actions=['emotion'],
                enforce_detection=False
            )
            
            # Extract emotion probabilities
            if isinstance(result, list):
                emotions = result[0]['emotion']
            else:
                emotions = result['emotion']
            
            # Map emotions to stress indicators
            stress_emotions = ['angry', 'fear', 'sad']
            non_stress_emotions = ['happy', 'surprise', 'neutral']
            
            stress_score = sum(emotions.get(emotion, 0) for emotion in stress_emotions)
            non_stress_score = sum(emotions.get(emotion, 0) for emotion in non_stress_emotions)
            
            # Add disgust as neutral (not clearly stress or non-stress)
            disgust_score = emotions.get('disgust', 0)
            non_stress_score += disgust_score * 0.5
            stress_score += disgust_score * 0.5
            
            # Normalize to 100%
            total = stress_score + non_stress_score
            if total > 0:
                stress_percentage = (stress_score / total) * 100
                non_stress_percentage = (non_stress_score / total) * 100
            else:
                stress_percentage = 50.0
                non_stress_percentage = 50.0
            
            deepface_result = {
                'stress': round(stress_percentage, 2),
                'non_stress': round(non_stress_percentage, 2),
                'dominant_emotion': max(emotions, key=emotions.get),
                'dominant_state': 'stress' if stress_percentage > non_stress_percentage else 'non_stress',
                'confidence': round(max(stress_percentage, non_stress_percentage), 2),
                'source': 'facial_emotion',
                'raw_emotions': emotions
            }
            
            return deepface_result
            
        except Exception as e:
            return {"error": f"Facial emotion analysis failed: {e}"}
    
    def multimodal_prediction(self, signal_sequence=None, image=None, fusion_method='weighted_average'):
        """
        Combine physiological and facial emotion predictions
        """
        results = {}
        
        # Get physiological prediction
        if signal_sequence is not None:
            physio_result = self.predict_physiological_stress(signal_sequence)
            results['physiological'] = physio_result
        
        # Get facial emotion prediction
        if image is not None:
            facial_result = self.analyze_facial_emotion(image)
            results['facial'] = facial_result
        
        # Fusion of results
        if len(results) > 1 and all('error' not in r for r in results.values()):
            if fusion_method == 'weighted_average':
                # Weight physiological signals more heavily (0.7) than facial (0.3)
                physio_weight = 0.7
                facial_weight = 0.3
                
                physio_stress = results['physiological']['stress'] / 100
                facial_stress = results['facial']['stress'] / 100
                
                fused_stress = (physio_stress * physio_weight + facial_stress * facial_weight)
                fused_non_stress = 1 - fused_stress
                
                fused_result = {
                    'stress': round(fused_stress * 100, 2),
                    'non_stress': round(fused_non_stress * 100, 2),
                    'dominant_state': 'stress' if fused_stress > 0.5 else 'non_stress',
                    'confidence': round(max(fused_stress, fused_non_stress) * 100, 2),
                    'source': 'multimodal_fusion',
                    'fusion_method': fusion_method,
                    'weights': {'physiological': physio_weight, 'facial': facial_weight}
                }
                
                results['fused'] = fused_result
        
        return results

def demo_multimodal_detection():
    """Demonstrate the multimodal stress detection system"""
    
    print("🚀 MULTIMODAL STRESS DETECTION DEMO")
    print("=" * 50)
    
    # Initialize detector
    detector = MultimodalStressDetector()
    
    # Demo 1: Physiological signal prediction (synthetic data)
    print("\n📊 Demo 1: Physiological Signal Analysis")
    print("-" * 30)
    
    if detector.physio_model is not None:
        # Create synthetic physiological data (240 time steps, multiple channels)
        # This simulates a window of multimodal physiological data
        synthetic_signals = np.random.randn(240, 10)  # 240 timesteps, 10 signal channels
        
        physio_result = detector.predict_physiological_stress(synthetic_signals)
        print("Physiological Analysis Result:")
        for key, value in physio_result.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Physiological model not available - train the model first!")
    
    # Demo 2: Facial emotion analysis (create a test image)
    print("\n😊 Demo 2: Facial Emotion Analysis")
    print("-" * 30)
    
    try:
        # Create a synthetic face image (for demo purposes)
        test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        facial_result = detector.analyze_facial_emotion(test_image)
        print("Facial Emotion Analysis Result:")
        if 'error' not in facial_result:
            for key, value in facial_result.items():
                if key != 'raw_emotions':
                    print(f"  {key}: {value}")
        else:
            print(f"  Error: {facial_result['error']}")
            
    except Exception as e:
        print(f"❌ Facial analysis demo failed: {e}")
    
    # Demo 3: Multimodal fusion
    print("\n🔗 Demo 3: Multimodal Fusion")
    print("-" * 30)
    
    if detector.physio_model is not None:
        try:
            multimodal_result = detector.multimodal_prediction(
                signal_sequence=synthetic_signals,
                image=test_image
            )
            
            print("Multimodal Analysis Results:")
            for source, result in multimodal_result.items():
                print(f"\n{source.upper()} Results:")
                if 'error' not in result:
                    for key, value in result.items():
                        if key not in ['raw_emotions', 'weights']:
                            print(f"  {key}: {value}")
                else:
                    print(f"  Error: {result['error']}")
                    
        except Exception as e:
            print(f"❌ Multimodal fusion demo failed: {e}")
    
    print("\n✨ Demo completed!")
    print("\n📝 Notes:")
    print("- Train the CNN+LSTM model first using train_stress_nn.py")
    print("- For real applications, use actual physiological data and face images")
    print("- The system combines multiple modalities for robust stress detection")

if __name__ == "__main__":
    demo_multimodal_detection()
