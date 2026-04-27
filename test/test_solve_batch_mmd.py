import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from filter_data import solve_mmd, solve_batch_mmd, compute_kernel

def test_solve_batch_mmd_vs_solve_mmd():
    np.random.seed(42)
    batch_size = 5
    max_seq_len = 8
    dim = 3

    # Generate target data
    tar_data = np.random.randn(20, dim)
    tar_kernel = compute_kernel(tar_data, kernel_type='rbf')
    tar_kernel_mean = tar_kernel.mean()

    # Generate batch of source sequences with padding (zero rows at the start)
    src_batch_data = np.zeros((batch_size, max_seq_len, dim))
    true_lengths = np.random.randint(3, max_seq_len + 1, size=batch_size)
    for i in range(batch_size):
        seq_len = true_lengths[i]
        src_batch_data[i, max_seq_len - seq_len:] = np.random.randn(seq_len, dim)

    # Compute batch MMD costs
    mmd_costs = solve_batch_mmd(src_batch_data, tar_data, tar_kernel_mean, kernel_type='rbf')

    # Compare with solve_mmd for each sequence (ignoring padding)
    for i in range(batch_size):
        seq = src_batch_data[i]
        # Remove padding (rows that are all zeros at the start)
        padding_len = 0
        for j in range(max_seq_len):
            if np.all(seq[j] == 0):
                padding_len += 1
            else:
                break
        seq_unpadded = seq[padding_len:]
        mmd_single = solve_mmd(seq_unpadded, tar_data, tar_kernel, kernel_type='rbf')
        assert np.isclose(mmd_costs[i], mmd_single, atol=1e-6), f"Mismatch at batch {i}: {mmd_costs[i]} vs {mmd_single}"
    print("All solve_batch_mmd results match solve_mmd.")

if __name__ == "__main__":
    test_solve_batch_mmd_vs_solve_mmd()
