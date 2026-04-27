import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import h5py
import d4rl
import random
from tqdm import tqdm, trange
from sklearn.metrics.pairwise import rbf_kernel
from filter_data import make_tensor_list_from_buffer
from decision_transformer.misc.utils import TrajectoryBuffer, call_d4rl_dataset, call_tar_dataset

class TestMMDCosts(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment by loading the pre-computed cost data and
        reconstructing the source buffer.
        """
        self.env = 'hopper'
        self.srctype = 'medium'
        self.tartype = 'kinematic_medium'
        self.seq_len = 20

        self.src_dataset_name = f'{self.env}-{self.srctype}-v2'
        self.tar_dataset_name = f'{self.env}-{self.tartype}-v2'

        # Load the source dataset to reconstruct the buffer
        src_dataset = call_d4rl_dataset(self.src_dataset_name)
        # Reconstruct the source buffer exactly as in filter_data.py
        self.src_buffer = TrajectoryBuffer(src_dataset, seq_len=self.seq_len, padding='left')
        self.num_trajectories = len(self.src_buffer.observations)
        self.masks = self.src_buffer.masks[:self.num_trajectories]
        # Load target dataset and create buffer
        tar_dataset = call_tar_dataset('data/target_dataset', self.env, self.tartype)
        self.tar_buffer = TrajectoryBuffer(tar_dataset, seq_len=self.seq_len, padding='left')

        # Construct the output file path
        self.output_file = f"data/costs/{self.env}-srcdatatype-{self.srctype}-tardatatype-{self.tartype}.hdf5"

        # Skip tests if the data file doesn't exist
        if not os.path.exists(self.output_file):
            self.skipTest(f"Test data file not found: {self.output_file}. "
                          f"Generate it by running: python filter_data.py --env {self.env} "
                          f"--srctype {self.srctype} --tartype {self.tartype} --seq_len {self.seq_len}")
        # Load the pre-computed costs
        with h5py.File(self.output_file, 'r') as hfile:
            self.mmd_cost = hfile['mmd_cost'][:]
        
        print(f"Loaded source dataset '{self.src_dataset_name}' with {self.num_trajectories} trajectories.")
        # for k, v in self.src_buffer.__dict__.items():
        #     print(k, v.shape if isinstance(v, np.ndarray) else v)
        print(f"Loaded target dataset '{self.tar_dataset_name}' with {len(self.tar_buffer.observations)} trajectories.")
        # for k, v in self.tar_buffer.__dict__.items():
        #     print(k, v.shape if isinstance(v, np.ndarray) else v)
        print(f"Loaded MMD costs from {self.output_file} with shape {self.mmd_cost.shape}")

        valid_indices = np.nonzero(self.src_buffer.masks[:self.num_trajectories])
        valid_indices = np.array(list(zip(*valid_indices)), dtype=np.int32)

        # Select a small number of random trajectories to test for efficiency
        num_samples = 100
        # sampled_indices = random.sample(range(len(valid_indices)), min(num_samples, len(valid_indices)))
        sampled_indices = np.random.choice(len(valid_indices), min(num_samples, len(valid_indices)), replace=False)
        if self.num_trajectories > 0:
            self.test_indices = valid_indices[sampled_indices]
        else:
            self.test_indices = []
        print(f"Selected {len(self.test_indices)} random trajectories for testing.")
        print("Test indices:", self.test_indices)

    def test_mmd_cost_shape_is_correct(self):
        """
        Verify that the shape of the mmd_cost array matches the expected
        (num_trajectories, max_len) from the source buffer.
        """
        expected_shape = (self.num_trajectories, self.src_buffer.max_len)
        self.assertEqual(self.mmd_cost.shape, expected_shape,
                         f"Shape mismatch: Got {self.mmd_cost.shape}, expected {expected_shape}")

    def test_mmd_cost_values_are_negative(self):
        """
        Verify that all valid (non-padded) MMD cost values are negative,
        as they are negated before being saved.
        """
        valid_costs = self.mmd_cost[self.masks == 1]
        self.assertTrue(np.all(valid_costs <= 0),
                        "Not all valid MMD costs are negative.")

    def test_mmd_cost_masking_is_correct(self):
        """
        Verify that costs in padded regions are -inf and costs in valid
        regions are finite.
        """
        # Check that padded regions have -inf cost
        padded_costs = self.mmd_cost[self.masks == 0]
        self.assertTrue(np.all(padded_costs == -np.inf),
                        "Padded regions do not have -inf cost.")

        # Check that valid regions have finite costs
        valid_costs = self.mmd_cost[self.masks == 1]
        self.assertTrue(np.all(np.isfinite(valid_costs)),
                        "Valid regions contain non-finite costs.")
        
    def test_mmd_costs_match_precomputed(self):
        """
        Verify that the MMD costs match the pre-computed values for a small
        number of random trajectories.
        """
        cost_file_mmd_costs = self.mmd_cost[self.test_indices[:, 0], self.test_indices[:, 1]]
        recalculated_costs = np.full(len(self.test_indices), np.inf, dtype=np.float32)
        for t, idx in tqdm(enumerate(self.test_indices), total=len(self.test_indices)):
            src_start = idx[1]
            src_end = src_start + self.seq_len
            if self.src_buffer.padding == 'left':
                src_end = min(src_end, self.src_buffer.max_len)
            else:
                src_end = min(src_end, self.src_buffer.traj_lens[idx[0]])
            # Extract the source sequence
            src_arr = np.hstack([
                self.src_buffer.observations[idx[0]][src_start:src_end],
                self.src_buffer.actions[idx[0]][src_start:src_end],
                self.src_buffer.next_observations[idx[0]][src_start:src_end],
                self.src_buffer.rewards[idx[0]][src_start:src_end].reshape(-1, 1),
            ])
            # iterate through all possible target sequences
            gamma = 1.0/src_arr.shape[1] # Adjust gamma based on feature dimension
            src_kernel = rbf_kernel(src_arr, gamma=gamma)
            # print(f"src_arr shape: {src_arr.shape}" )
            # print(f"tar_buffer observations info: {len(self.tar_buffer.observations)} episodes, {self.tar_buffer.observations[0].shape} shape")

            for i in range(len(self.tar_buffer.observations)):
                for tar_start in np.nonzero(self.tar_buffer.masks[i])[0]:
                    tar_end = tar_start + self.seq_len
                    if self.tar_buffer.padding == 'left':
                        tar_end = min(tar_end, self.tar_buffer.max_len)
                    else:
                        tar_end = min(tar_end, self.tar_buffer.traj_lens[i])
                    tar_arr = np.hstack([
                        self.tar_buffer.observations[i][tar_start:tar_end],
                        self.tar_buffer.actions[i][tar_start:tar_end],
                        self.tar_buffer.next_observations[i][tar_start:tar_end],
                        self.tar_buffer.rewards[i][tar_start:tar_end].reshape(-1, 1),
                    ])
                    # Compute the MMD cost
                    tar_kernel = rbf_kernel(tar_arr, gamma=gamma)
                    cross_kernel = rbf_kernel(src_arr, tar_arr, gamma=gamma)
                    mmd_cost = np.mean(src_kernel) + np.mean(tar_kernel) - 2 * np.mean(cross_kernel)
                    recalculated_costs[t] = min(recalculated_costs[t], mmd_cost)
        recalculated_costs = -recalculated_costs
        print(f"Pre-computed costs: {cost_file_mmd_costs}")
        print(f"Recalculated costs: {recalculated_costs}")
        self.assertTrue(np.allclose(cost_file_mmd_costs, recalculated_costs,
                                   rtol=1e-5, atol=1e-8),
                        "MMD costs do not match pre-computed values.")
        print("All MMD costs match pre-computed values.")



if __name__ == "__main__":
    unittest.main()