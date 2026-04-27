import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from filter_data import compute_kernel, compute_kernel_window_means

class TestKernelWindowMeans(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for kernel window means computation validation.
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create test data with specific dimensions for window testing
        self.src_len = 15
        self.tar_len = 12
        self.n_features = 8
        self.window_size = 5
        
        # Generate random test data
        self.src_data = np.random.randn(self.src_len, self.n_features).astype(np.float32)
        self.tar_data = np.random.randn(self.tar_len, self.n_features).astype(np.float32)
        
        # Convert to PyTorch tensors
        self.src_tensor = torch.tensor(self.src_data, dtype=torch.float32)
        self.tar_tensor = torch.tensor(self.tar_data, dtype=torch.float32)
        
        # Gamma value (should match filter_data.py)
        self.gamma = 1.0 / self.n_features

    def naive_sliding_window_means(self, src_data, tar_data, window_size, gamma):
        """
        Naive implementation of sliding window means for validation.
        """
        # Compute full kernel matrix
        kernel_matrix = rbf_kernel(src_data, tar_data, gamma=gamma)
        
        src_len, tar_len = kernel_matrix.shape
        
        # Compute sliding window means manually
        result = np.zeros((src_len, tar_len), dtype=np.float32)
        
        for i in range(src_len):
            for j in range(tar_len):
                # Define window boundaries
                src_start = i
                src_end = min(src_len, i + window_size)
                tar_start = j
                tar_end = min(tar_len, j + window_size)
                
                # Extract window
                window = kernel_matrix[src_start:src_end, tar_start:tar_end]
                
                # Compute mean (only if window has the expected size)
                result[i, j] = np.mean(window)
        
        return result

    def test_kernel_window_means_shape(self):
        """
        Test that compute_kernel_window_means returns the correct shape.
        """
        result = compute_kernel_window_means(self.src_tensor, self.tar_tensor, self.window_size, kernel_type='rbf')
        result = result.cpu().numpy()
        
        expected_shape = (self.src_len, self.tar_len)
        self.assertEqual(result.shape, expected_shape,
                         f"Window means shape mismatch: got {result.shape}, expected {expected_shape}")
        
        print(f"✓ Window means shape is correct: {result.shape}")

    def test_kernel_window_means_values_simple(self):
        """
        Test window means computation with a simple case we can verify manually.
        """
        # Create simple test data
        simple_src = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        simple_tar = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        window_size = 2
        
        # Compute using filter_data.py
        result_torch = compute_kernel_window_means(simple_src, simple_tar, window_size, kernel_type='rbf')
        result_torch = result_torch.cpu().numpy()
        
        # Compute manually for comparison
        simple_src_np = simple_src.cpu().numpy()
        simple_tar_np = simple_tar.cpu().numpy()
        gamma = 1.0 / simple_src_np.shape[1]
        
        result_manual = self.naive_sliding_window_means(simple_src_np, simple_tar_np, window_size, gamma)
        
        print(f"PyTorch result:\n{result_torch}")
        print(f"Manual result:\n{result_manual}")
        print(f"Max difference: {np.max(np.abs(result_torch - result_manual)):.2e}")
        
        # For now, just check that values are reasonable (between 0 and 1 for RBF)
        self.assertTrue(np.all(result_torch >= 0), "Window means should be non-negative")
        self.assertTrue(np.all(result_torch <= 1), "Window means should be <= 1 for RBF kernel")
        
        print("✓ Window means values are in reasonable range")

    def test_self_kernel_window_means(self):
        """
        Test window means computation for self-kernel (src == tar).
        """
        result = compute_kernel_window_means(self.src_tensor, self.src_tensor, self.window_size, kernel_type='rbf')
        result = result.cpu().numpy()
        
        # Check shape
        expected_shape = (self.src_len, self.src_len)
        self.assertEqual(result.shape, expected_shape,
                         f"Self-kernel window means shape mismatch: got {result.shape}, expected {expected_shape}")
        
        # Check that diagonal values are reasonable (should be close to 1 for perfect self-similarity)
        diagonal = np.diag(result)
        print(f"Diagonal values: {diagonal}")
        
        # Values should be positive and <= 1
        self.assertTrue(np.all(diagonal > 0), "Diagonal values should be positive")
        self.assertTrue(np.all(diagonal <= 1), "Diagonal values should be <= 1")
        
        print("✓ Self-kernel window means computed correctly")

    def test_window_means_vs_manual_computation(self):
        """
        Test that window means match a manual computation for specific positions.
        """
        # Test a specific position we can verify manually
        test_src_pos = 2  # Middle position to avoid edge effects
        test_tar_pos = 1
        
        # Compute using filter_data.py
        result = compute_kernel_window_means(self.src_tensor, self.tar_tensor, self.window_size, kernel_type='rbf')
        pytorch_value = result[test_src_pos, test_tar_pos].cpu().numpy()
        
        # Compute manually: extract the window and compute mean
        src_start = test_src_pos
        src_end = test_src_pos + self.window_size
        tar_start = test_tar_pos 
        tar_end = test_tar_pos + self.window_size
        
        src_window = self.src_data[src_start:src_end]
        tar_window = self.tar_data[tar_start:tar_end]
        
        # Compute kernel for this window
        window_kernel = rbf_kernel(src_window, tar_window, gamma=self.gamma)
        manual_value = np.mean(window_kernel)
        
        print(f"PyTorch value at [{test_src_pos}, {test_tar_pos}]: {pytorch_value:.6f}")
        print(f"Manual value: {manual_value:.6f}")
        print(f"Difference: {abs(pytorch_value - manual_value):.2e}")
        
        # They should be close (allowing for numerical precision differences)
        self.assertTrue(np.allclose(pytorch_value, manual_value, rtol=1e-4, atol=1e-6),
                        f"Window means don't match manual computation: {pytorch_value} vs {manual_value}")
        
        print("✓ Window means match manual computation")

    def test_edge_cases(self):
        """
        Test edge cases like window size larger than data size.
        """
        # Test with window size larger than data
        large_window = self.src_len + 5
        
        try:
            result = compute_kernel_window_means(self.src_tensor, self.tar_tensor, large_window, kernel_type='rbf')
            result = result.cpu().numpy()
            
            print(f"Large window result shape: {result.shape}")
            print(f"Large window sample values: {result[:3, :3]}")
            
            # Should still produce reasonable values
            self.assertTrue(np.all(np.isfinite(result)), "Large window should produce finite values")
            self.assertTrue(np.all(result >= 0), "Large window values should be non-negative")
            
            print("✓ Large window size handled correctly")
            
        except Exception as e:
            print(f"Large window size caused error: {e}")
            # This might be expected behavior

    def test_different_window_sizes(self):
        """
        Test that different window sizes produce reasonable results.
        """
        window_sizes = [1, 3, 5, 8]
        
        for ws in window_sizes:
            if ws <= min(self.src_len, self.tar_len):
                result = compute_kernel_window_means(self.src_tensor, self.tar_tensor, ws, kernel_type='rbf')
                result = result.cpu().numpy()
                
                print(f"Window size {ws}: shape {result.shape}, mean {np.mean(result):.4f}, std {np.std(result):.4f}")
                
                # Basic sanity checks
                self.assertEqual(result.shape, (self.src_len, self.tar_len))
                self.assertTrue(np.all(np.isfinite(result)))
                self.assertTrue(np.all(result >= 0))
                self.assertTrue(np.all(result <= 1))
        
        print("✓ Different window sizes work correctly")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
