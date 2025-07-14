"""
Unit tests for mode crossover validation functionality.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the code directory to the path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
code_dir = os.path.join(base_dir, 'code')
sys.path.insert(0, base_dir)
sys.path.insert(0, code_dir)

from model.adaptability_model import AdaptabilityModel
from utils.computational_utils import find_mode_crossover
from validation.validate_theory import validate_mode_crossovers


class TestModeCrossovers(unittest.TestCase):
    """Test cases for mode crossover detection and validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_12 = AdaptabilityModel([1, 2])
        self.model_123 = AdaptabilityModel([1, 2, 3])

    def test_validate_mode_crossovers_returns_dataframe(self):
        """Test that validate_mode_crossovers returns a DataFrame."""
        result = validate_mode_crossovers()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Crossover', result.columns)

    def test_validate_mode_crossovers_finds_expected_crossover(self):
        """Test that validate_mode_crossovers finds the expected crossover point."""
        result = validate_mode_crossovers()
        
        # Should find at least one crossover
        self.assertGreater(len(result), 0)
        
        # The crossover should be near the theoretical value x_c ≈ 0.1762
        crossovers = result['Crossover'].values
        expected_crossover = 0.1762
        
        # Check if any crossover is within reasonable tolerance of expected value
        found_expected = any(abs(x - expected_crossover) < 0.01 for x in crossovers)
        self.assertTrue(found_expected, 
                       f"Expected crossover near {expected_crossover}, found {crossovers}")

    def test_find_mode_crossover_basic_functionality(self):
        """Test basic functionality of find_mode_crossover."""
        crossovers = find_mode_crossover(self.model_12, 1, 2, x_range=(0, 0.25))
        
        # Should find at least one crossover
        self.assertGreater(len(crossovers), 0)
        
        # All crossovers should be within the specified range
        for x_c in crossovers:
            self.assertGreaterEqual(x_c, 0)
            self.assertLessEqual(x_c, 0.25)

    def test_crossover_point_accuracy(self):
        """Test that crossover points are accurate (modes have equal values)."""
        crossovers = find_mode_crossover(self.model_12, 1, 2, x_range=(0, 0.25))
        
        for x_c in crossovers:
            # At crossover point, M_1(x_c) should equal M_2(x_c)
            m1_value = self.model_12.compute_mode_decay(1, x_c)
            m2_value = self.model_12.compute_mode_decay(2, x_c)
            
            # Allow small numerical tolerance
            self.assertAlmostEqual(m1_value, m2_value, places=4,
                                 msg=f"At x_c={x_c}, M_1={m1_value}, M_2={m2_value}")

    def test_crossover_detection_different_ranges(self):
        """Test crossover detection in different x ranges."""
        # Test narrow range around expected crossover
        narrow_crossovers = find_mode_crossover(self.model_12, 1, 2, x_range=(0.15, 0.2))
        
        # Test wider range
        wide_crossovers = find_mode_crossover(self.model_12, 1, 2, x_range=(0, 0.5))
        
        # Narrow range should find fewer or equal crossovers
        self.assertLessEqual(len(narrow_crossovers), len(wide_crossovers))

    def test_no_crossover_in_inappropriate_range(self):
        """Test that no crossovers are found in ranges where they shouldn't exist."""
        # Test a range far from the expected crossover
        far_crossovers = find_mode_crossover(self.model_12, 1, 2, x_range=(0.4, 0.5))
        
        # Should find no crossovers in this range
        self.assertEqual(len(far_crossovers), 0)

    def test_validate_mode_crossovers_output_format(self):
        """Test the output format of validate_mode_crossovers."""
        # Capture stdout to test print statements
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            result = validate_mode_crossovers()
        
        output = f.getvalue()
        
        # Check that validation header is printed
        self.assertIn("Validating Mode Crossovers", output)
        
        # Check that crossover detection is reported
        self.assertIn("Detected crossover", output)

    def test_crossover_reproducibility(self):
        """Test that crossover detection is reproducible."""
        crossovers1 = find_mode_crossover(self.model_12, 1, 2, x_range=(0, 0.25))
        crossovers2 = find_mode_crossover(self.model_12, 1, 2, x_range=(0, 0.25))
        
        # Results should be identical
        np.testing.assert_array_almost_equal(crossovers1, crossovers2)


if __name__ == '__main__':
    unittest.main()
