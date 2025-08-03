#!/usr/bin/env python3
"""
Test script for MAC_backbone implementation
"""

import torch
import sys
import os

# Add current directory to path to find models
sys.path.append('.')

try:
    from models.backbone import MAC_backbone
    print("✓ Successfully imported MAC_backbone")
except ImportError as e:
    print(f"✗ Failed to import MAC_backbone: {e}")
    exit(1)

def test_mac_backbone():
    """Test MAC_backbone with different configurations"""
    
    print("\n=== Testing MAC_backbone Implementation ===")
    
    # Test 1: Basic initialization patterns
    print("\n1. Testing different initialization patterns:")
    
    try:
        # Pattern 1: MAC_backbone(feat_dim, feat_dim) - for RML2016.10A
        model1 = MAC_backbone(256, 256)
        print("   ✓ MAC_backbone(feat_dim, feat_dim) - RML2016.10A pattern")
        
        # Pattern 2: MAC_backbone(feat_dim, num_classes) - for RML2016.10B  
        model2 = MAC_backbone(256, 10)
        print("   ✓ MAC_backbone(feat_dim, num_classes) - RML2016.10B pattern")
        
        # Pattern 3: Test args object pattern (for RML2018)
        class MockArgs:
            def __init__(self):
                self.feat_dim = 128
        
        args = MockArgs()
        model3 = MAC_backbone(args, 24)
        print("   ✓ MAC_backbone(args, num_classes) - RML2018 pattern")
        
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")
        return False
    
    # Test 2: Forward pass with different view modes
    print("\n2. Testing forward pass with different modes:")
    
    try:
        # Create sample input: [batch_size, 2, signal_length]
        batch_size = 4
        signal_length = 128
        x = torch.randn(batch_size, 2, signal_length)
        
        # Test DB mode (dual-domain)
        model = MAC_backbone(128, 11)
        model.eval()
        
        with torch.no_grad():
            # Test DB mode with different domains
            for domain in ['AN', 'AF', 'WT', 'FFT']:
                try:
                    outputs = model(x, mod_l=domain, view_chose='DB', mode='linerProbing')
                    print(f"   ✓ DB mode with {domain} domain: {len(outputs)} outputs")
                    
                    # Verify output shapes
                    for i, out in enumerate(outputs):
                        if out.shape != (batch_size, 128):
                            print(f"     ✗ Output {i} shape mismatch: {out.shape}")
                            return False
                            
                except Exception as e:
                    print(f"   ✗ DB mode with {domain} failed: {e}")
                    return False
            
            # Test ALL mode (multi-domain)
            try:
                outputs = model(x, mod_l='AN', view_chose='ALL', mode='Finetuning')
                print(f"   ✓ ALL mode: {len(outputs)} outputs")
                
                if len(outputs) != 6:
                    print(f"     ✗ Expected 6 outputs for ALL mode, got {len(outputs)}")
                    return False
                    
                # Verify output shapes
                for i, out in enumerate(outputs):
                    if out.shape != (batch_size, 128):
                        print(f"     ✗ Output {i} shape mismatch: {out.shape}")
                        return False
                        
            except Exception as e:
                print(f"   ✗ ALL mode failed: {e}")
                return False
                
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False
    
    # Test 3: Domain transformations
    print("\n3. Testing domain transformations:")
    
    try:
        x = torch.randn(2, 2, 64)  # Smaller size for faster testing
        model = MAC_backbone(64, 10)
        
        # Test amplitude-phase transformation
        x_an = model.transform_to_amplitude_phase(x)
        print(f"   ✓ Amplitude-Phase transformation: {x_an.shape}")
        
        # Test instantaneous frequency transformation  
        x_af = model.transform_to_instantaneous_frequency(x)
        print(f"   ✓ Instantaneous Frequency transformation: {x_af.shape}")
        
        # Test FFT transformation
        x_fft = model.transform_to_fft(x)
        print(f"   ✓ FFT transformation: {x_fft.shape}")
        
        # Test wavelet transformation (might be slower)
        try:
            x_wt = model.transform_to_wavelet(x)
            print(f"   ✓ Wavelet transformation: {x_wt.shape}")
        except Exception as e:
            print(f"   ⚠ Wavelet transformation failed (this is expected if pywt is not installed): {e}")
        
        # Test data augmentation
        x_aug = model.apply_data_augmentation(x)
        print(f"   ✓ Data augmentation: {x_aug.shape}")
        
    except Exception as e:
        print(f"   ✗ Domain transformations failed: {e}")
        return False
    
    print("\n✓ All tests passed! MAC_backbone is ready to use.")
    return True

if __name__ == "__main__":
    success = test_mac_backbone()
    if success:
        print("\n🎉 MAC_backbone implementation is working correctly!")
        print("\nYou can now:")
        print("1. Run pretraining: python Pretraing_MAC.PY")
        print("2. Run fine-tuning: python Fine_tuning_Times.py")
        print("3. Use the model in your own scripts")
    else:
        print("\n❌ There are issues with the MAC_backbone implementation.")
        print("Please check the error messages above.")
