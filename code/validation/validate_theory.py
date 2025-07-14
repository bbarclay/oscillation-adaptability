"""
Validation script for the core theoretical findings in the adaptability model.

This script empirically verifies the key mathematical theorems from the paper by
running numerical tests on various model configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from pathlib import Path
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.adaptability_model import AdaptabilityModel
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from computational_utils import (
    verify_conservation_law,
    verify_exponential_decay,
    verify_temporal_oscillations,
    find_mode_crossover,
    verify_complexity_ceiling
)

# Ensure figures directory exists
FIGURES_DIR = Path('../../figures')
FIGURES_DIR.mkdir(exist_ok=True)


def validate_conservation_law():
    """
    Validate Theorem 3.1: Exact Additive Conservation C(x,d) + A(x,d) = 1.
    """
    print("\n==== Validating Theorem 3.1: Conservation Law C + A = 1 ====")

    # Test with different orbital order sets
    models = [
        ("Harmonic", AdaptabilityModel([1, 2, 3])),
        ("Odd Harmonic", AdaptabilityModel([1, 3, 5])),
        ("Mixed", AdaptabilityModel([2, 3, 5]))
    ]

    results = []

    for name, model in models:
        # Test over a wide range of points
        conservation_sums, max_deviation = verify_conservation_law(
            model, x_samples=200, d_samples=20,
            x_range=(-1, 1), d_range=(1, 30)
        )

        mean_sum = np.mean(conservation_sums)
        std_sum = np.std(conservation_sums)

        print(f"Model {name}:")
        print(f"  Mean C+A: {mean_sum:.16f}")
        print(f"  Standard deviation: {std_sum:.16e}")
        print(f"  Maximum absolute deviation from 1: {max_deviation:.16e}")

        results.append({
            'Model': name,
            'Mean C+A': mean_sum,
            'Std Dev': std_sum,
            'Max Deviation': max_deviation
        })

    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    print("\nSummary of Conservation Law Validation:")
    print(df.to_string(index=False))

    return df


def validate_exponential_decay():
    """
    Validate Theorem 3.3: Exponential Convergence of Adaptability.
    """
    print("\n==== Validating Theorem 3.3: Exponential Decay of Adaptability ====")

    # Test with different orbital order sets
    models = [
        ("Harmonic", AdaptabilityModel([1, 2, 3])),
        ("Odd Harmonic", AdaptabilityModel([1, 3, 5])),
        ("Mixed", AdaptabilityModel([2, 3, 5]))
    ]

    # Test configurations
    x_values = [0.125, 0.25, 0.375]
    d_range = (1, 30)

    results = []

    for name, model in models:
        for x in x_values:
            d_values, adaptability_values, fitted_exp, theoretical_exp, r_squared = (
                verify_exponential_decay(model, x, d_range, nd=100)
            )

            # Compute relative error
            if np.isnan(fitted_exp) or np.isnan(theoretical_exp):
                rel_error = np.nan
            else:
                rel_error = 100 * abs(fitted_exp - theoretical_exp) / abs(theoretical_exp)

            m_star, n_star = model.M_star(x)

            print(f"Model {name}, x = {x}:")
            print(f"  M* = {m_star:.6f}, N_ord* = {n_star}")
            print(f"  Theoretical exponent: {theoretical_exp:.6f}")
            print(f"  Fitted exponent: {fitted_exp:.6f}")
            print(f"  Relative error: {rel_error:.6f}%")
            print(f"  R² = {r_squared:.6f}")

            results.append({
                'Model': name,
                'x': x,
                'M*': m_star,
                'N_ord*': n_star,
                'Theoretical Exponent': theoretical_exp,
                'Fitted Exponent': fitted_exp,
                'Relative Error (%)': rel_error,
                'R²': r_squared
            })

    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    print("\nSummary of Exponential Decay Validation:")
    print(df.to_string(index=False))

    return df


def validate_temporal_oscillations():
    """
    Validate Theorem 4.1: Oscillation Necessity in Time and conservation in the time-dependent model.
    """
    print("\n==== Validating Theorem 4.1: Necessary Oscillations in Time ====")

    # Test with different orbital order sets
    models = [
        ("Harmonic", AdaptabilityModel([1, 2, 3])),
        ("Odd Harmonic", AdaptabilityModel([1, 3, 5])),
        ("Mixed", AdaptabilityModel([2, 3, 5]))
    ]

    # Test at various depths
    depths = [5, 10, 15, 20, 25]

    results = []

    for name, model in models:
        print(f"\nModel {name}:")

        for d in depths:
            # Use x=0.25 as a representative point
            x = 0.25

            max_deviation, oscillation_amplitude, mean_adaptability = (
                verify_temporal_oscillations(model, x, d, (0, 100), 1000)
            )

            print(f"  Depth {d}:")
            print(f"    Max deviation from C+A=1: {max_deviation:.2e}")
            print(f"    Oscillation amplitude: {oscillation_amplitude:.4f}")
            print(f"    Mean adaptability: {mean_adaptability:.4f}")

            results.append({
                'Model': name,
                'Depth': d,
                'Max Deviation': max_deviation,
                'Oscillation Amplitude': oscillation_amplitude,
                'Mean Adaptability': mean_adaptability
            })

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    print("\nSummary of Temporal Oscillations Validation:")
    print(df.to_string(index=False))

    # Calculate amplitude decay rate with depth
    for name in set(df['Model']):
        model_data = df[df['Model'] == name]
        depths = model_data['Depth'].values
        amplitudes = model_data['Oscillation Amplitude'].values

        # Fit exponential decay to amplitudes
        if len(depths) > 1:
            log_amplitudes = np.log(amplitudes)
            coeffs = np.polyfit(depths, log_amplitudes, 1)
            amplitude_decay_rate = coeffs[0]

            print(f"\nModel {name} - Amplitude decay with depth:")
            print(f"  Amplitude ∝ e^({amplitude_decay_rate:.6f} * d)")

    return df


def validate_mode_crossovers():
    """Validate existence of mode crossovers and locate critical points."""
    print("\n==== Validating Mode Crossovers and Critical Points ====")

    model = AdaptabilityModel([1, 2])
    crossovers = find_mode_crossover(model, 1, 2, x_range=(0, 0.25))

    for x_c in crossovers:
        print(f"  Detected crossover near x = {x_c:.6f}")

    return pd.DataFrame({'Crossover': crossovers})


def validate_complexity_ceiling():
    """Validate absence of triple-mode resonances (complexity ceiling)."""
    print("\n==== Validating Complexity Ceiling ====")

    model = AdaptabilityModel([1, 2, 3])
    triple_points = verify_complexity_ceiling(model)

    if triple_points:
        print(f"  Triple-mode resonance detected at {triple_points}")
    else:
        print("  No triple-mode resonance found (complexity ceiling observed).")

    return pd.DataFrame({'TriplePoint': triple_points})


def run_all_validations():
    """Run all validation tests and return compiled results."""
    print("===== STARTING COMPREHENSIVE MODEL VALIDATION =====")
    start_time = time.time()

    conservation_results = validate_conservation_law()
    exponential_decay_results = validate_exponential_decay()
    temporal_oscillation_results = validate_temporal_oscillations()
    crossover_results = validate_mode_crossovers()
    ceiling_results = validate_complexity_ceiling()

    end_time = time.time()
    print(f"\n===== VALIDATION COMPLETE =====")
    print(f"Total validation time: {end_time - start_time:.2f} seconds")

    return {
        'conservation': conservation_results,
        'exponential_decay': exponential_decay_results,
        'temporal_oscillations': temporal_oscillation_results,
        'mode_crossovers': crossover_results,
        'complexity_ceiling': ceiling_results
    }


if __name__ == "__main__":
    results = run_all_validations()

    # Save results to CSV files
    output_dir = Path('../../validation_results')
    output_dir.mkdir(exist_ok=True)

    for name, df in results.items():
        df.to_csv(output_dir / f"{name}_validation.csv", index=False)
        print(f"Saved {name} validation results to {output_dir / f'{name}_validation.csv'}")