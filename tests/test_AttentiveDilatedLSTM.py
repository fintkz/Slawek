import torch
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.AttentiveDilatedLSTM import (
    AttentiveDilatedLSTM,
)  # Assuming we saved the previous code in model.py


class TestAttentiveDilatedLSTM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_length = 24
        self.input_size = 10
        self.hidden_size = 20
        self.attention_size = 15
        self.dilations = [1, 2, 4]  # Three layers with increasing dilation

        self.model = AttentiveDilatedLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            attention_size=self.attention_size,
            dilations=self.dilations,
        )

    def test_output_shape(self):
        # Test with batch_first=True (default)
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output, (h_n, c_n) = self.model(x)

        # Check output shapes
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_length, self.hidden_size)
        )
        self.assertEqual(len(h_n), len(self.dilations))
        self.assertEqual(len(c_n), len(self.dilations))

        for h, c in zip(h_n, c_n):
            self.assertEqual(h.shape, (self.batch_size, self.hidden_size))
            self.assertEqual(c.shape, (self.batch_size, self.hidden_size))

    def test_batch_first_flag(self):
        # Create model with batch_first=False
        model_seq_first = AttentiveDilatedLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            attention_size=self.attention_size,
            dilations=self.dilations,
            batch_first=False,
        )

        # Test both input formats
        x_batch_first = torch.randn(
            self.batch_size, self.seq_length, self.input_size
        )
        x_seq_first = x_batch_first.transpose(0, 1)

        # Set both models to eval mode and use the same random seed
        torch.manual_seed(42)
        self.model.eval()
        model_seq_first.eval()

        # Ensure the models have the same parameters
        for p1, p2 in zip(
            self.model.parameters(), model_seq_first.parameters()
        ):
            p2.data.copy_(p1.data)

        with torch.no_grad():
            output_batch_first, _ = self.model(x_batch_first)
            output_seq_first, _ = model_seq_first(x_seq_first)

        # Compare shapes first
        print(f"\nBatch-first output shape: {output_batch_first.shape}")
        print(f"Sequence-first output shape: {output_seq_first.shape}")

        # Transpose sequence-first output to match batch-first
        output_seq_first_transformed = output_seq_first.transpose(0, 1)

        # Compare the absolute difference
        max_diff = torch.max(
            torch.abs(output_batch_first - output_seq_first_transformed)
        )
        mean_diff = torch.mean(
            torch.abs(output_batch_first - output_seq_first_transformed)
        )
        print(f"Maximum absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")

        # Original assertion with increased tolerance
        self.assertTrue(
            torch.allclose(
                output_batch_first,
                output_seq_first_transformed,
                rtol=1e-4,
                atol=1e-4,
            ),
            "Outputs from batch-first and sequence-first modes don't match",
        )

    def test_dilation_pattern(self):
        # Small test to verify dilation pattern
        batch_size = 1
        seq_length = 8
        input_size = 1
        model = AttentiveDilatedLSTM(
            input_size=input_size,
            hidden_size=2,
            attention_size=2,
            dilations=[1, 2],  # Two layers with dilation 1 and 2
        )

        # Create input where each timestep is just its position number
        x = (
            torch.arange(seq_length)
            .float()
            .reshape(batch_size, seq_length, input_size)
        )
        output, _ = model(x)

        # The second layer (dilation=2) should only update on even timesteps
        # This is a simple heuristic test - the output at odd timesteps should be
        # more similar to previous timestep in the second layer
        second_layer_changes = torch.abs(output[0, 1:] - output[0, :-1])
        odd_timestep_changes = second_layer_changes[::2].mean()
        even_timestep_changes = second_layer_changes[1::2].mean()

        self.assertGreater(even_timestep_changes, odd_timestep_changes)


if __name__ == "__main__":
    unittest.main()
