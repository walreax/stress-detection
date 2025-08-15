import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np

def test_gpu_configuration():
    """Test GPU configuration and performance"""
    print("🔧 GPU CONFIGURATION TEST")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    print(f"\nBuilt with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    
    # List physical devices
    print(f"\nPhysical devices:")
    for device in tf.config.list_physical_devices():
        print(f"  {device}")
    
    # GPU-specific information
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"\n🚀 GPU DETAILS:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            
            # Get GPU details
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"    Compute capability: {gpu_details.get('compute_capability', 'Unknown')}")
            except:
                print(f"    Details: Not available")
        
        # Configure memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✅ Memory growth enabled for all GPUs")
        except RuntimeError as e:
            print(f"  ❌ Memory growth configuration failed: {e}")
        
        # Test mixed precision
        print(f"\n⚡ MIXED PRECISION TEST:")
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"  ✅ Mixed precision enabled: {policy.name}")
        except Exception as e:
            print(f"  ❌ Mixed precision failed: {e}")
    else:
        print(f"\n❌ No GPUs found!")
        return False
    
    # Performance test
    print(f"\n🏃 PERFORMANCE TEST:")
    
    # Create test tensors
    size = 2000
    
    # CPU test
    print(f"  Testing matrix multiplication ({size}x{size})...")
    with tf.device('/CPU:0'):
        cpu_start = tf.timestamp()
        a_cpu = tf.random.normal([size, size])
        b_cpu = tf.random.normal([size, size])
        c_cpu = tf.matmul(a_cpu, b_cpu)
        cpu_time = tf.timestamp() - cpu_start
        print(f"  CPU time: {cpu_time.numpy():.4f} seconds")
    
    # GPU test
    if gpus:
        with tf.device('/GPU:0'):
            gpu_start = tf.timestamp()
            a_gpu = tf.random.normal([size, size])
            b_gpu = tf.random.normal([size, size])
            c_gpu = tf.matmul(a_gpu, b_gpu)
            gpu_time = tf.timestamp() - gpu_start
            print(f"  GPU time: {gpu_time.numpy():.4f} seconds")
            
            speedup = cpu_time.numpy() / gpu_time.numpy()
            print(f"  🚀 GPU speedup: {speedup:.2f}x")
    
    # Memory test
    print(f"\n💾 MEMORY TEST:")
    try:
        if gpus:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            current_mb = memory_info['current'] / (1024**2)
            peak_mb = memory_info['peak'] / (1024**2)
            print(f"  Current GPU memory: {current_mb:.1f} MB")
            print(f"  Peak GPU memory: {peak_mb:.1f} MB")
    except Exception as e:
        print(f"  Memory info not available: {e}")
    
    # Test neural network creation
    print(f"\n🧠 NEURAL NETWORK TEST:")
    try:
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile with mixed precision if available
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Test forward pass
            test_input = tf.random.normal([32, 100])
            test_output = model(test_input)
            
            print(f"  ✅ Model created successfully on {'/GPU:0' if gpus else '/CPU:0'}")
            print(f"  Model output shape: {test_output.shape}")
            print(f"  Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}")
            
    except Exception as e:
        print(f"  ❌ Neural network test failed: {e}")
    
    print(f"\n✨ GPU configuration test completed!")
    return len(gpus) > 0

if __name__ == "__main__":
    gpu_available = test_gpu_configuration()
    
    if gpu_available:
        print(f"\n🎉 SUCCESS: GPU is properly configured and ready for training!")
        print(f"💡 Recommendations:")
        print(f"   - Use batch sizes of 64-128 for optimal GPU utilization")
        print(f"   - Enable mixed precision for faster training")
        print(f"   - Monitor GPU memory usage during training")
    else:
        print(f"\n⚠️  WARNING: No GPU detected. Training will use CPU.")
        print(f"💡 Recommendations:")
        print(f"   - Use smaller batch sizes (16-32)")
        print(f"   - Consider reducing model complexity")
        print(f"   - Expect longer training times")
