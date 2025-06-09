import pytest
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to the path to import AdaptabilityModel
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from adaptability_model import AdaptabilityModel

# Add the visualization directory to the path to import plotting functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plotting import plot_adaptability_landscape

# Define a directory for saving test plots
TEST_FIGURES_DIR = "test_figures"

@pytest.fixture(scope="session", autouse=True)
def create_test_figures_dir():
    if not os.path.exists(TEST_FIGURES_DIR):
        os.makedirs(TEST_FIGURES_DIR)
    yield
    # Clean up: remove files after tests run.
    # for f in os.listdir(TEST_FIGURES_DIR):
    #     os.remove(os.path.join(TEST_FIGURES_DIR, f))
    # os.rmdir(TEST_FIGURES_DIR)
    # For now, let's not remove the directory to inspect plots if needed.

@pytest.fixture
def model_fixture():
    """Provides a default AdaptabilityModel instance for tests."""
    return AdaptabilityModel(n_ord=[1, 2, 3])

def test_plot_adaptability_landscape_runs(model_fixture):
    """Test that plot_adaptability_landscape runs without errors."""
    fig = plot_adaptability_landscape(model_fixture, x_range=(-1, 1), d_range=(1, 10))
    assert fig is not None
    plt.close(fig)

def test_plot_adaptability_landscape_elements(model_fixture):
    """Test that the plot has correct title and labels."""
    n_ord_expected = model_fixture.n_ord
    fig = plot_adaptability_landscape(model_fixture, x_range=(-1, 1), d_range=(1, 10), title="Custom Title")
    ax = fig.axes[0]
    assert ax.get_title() == "Custom Title"
    assert ax.get_xlabel() == "Configuration (x)"
    assert ax.get_ylabel() == "Depth (d)"
    plt.close(fig)

    # Test default title
    fig_default_title = plot_adaptability_landscape(model_fixture, x_range=(-1, 1), d_range=(1, 10))
    ax_default = fig_default_title.axes[0]
    assert ax_default.get_title() == f"Adaptability Landscape for N_ord = {n_ord_expected}"
    plt.close(fig_default_title)


def test_plot_adaptability_landscape_save(model_fixture):
    """Test that the plot can be saved to a file."""
    save_path = os.path.join(TEST_FIGURES_DIR, "adaptability_landscape.png")
    if os.path.exists(save_path):
        os.remove(save_path) # Ensure clean state

    fig = plot_adaptability_landscape(model_fixture, x_range=(-1, 1), d_range=(1, 10), save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0 # Check that file is not empty
    plt.close(fig)
    # os.remove(save_path) # Clean up

# Example of how to run tests from within the script (optional)
# if __name__ == "__main__":
# pytest.main([__file__])

from plotting import plot_time_series

def test_plot_time_series_runs(model_fixture):
    """Test that plot_time_series runs without errors."""
    fig = plot_time_series(model_fixture, x=0.5, d=5, t_range=(0, 10))
    assert fig is not None
    plt.close(fig)

def test_plot_time_series_elements(model_fixture):
    """Test that the plot has correct title, labels, and lines."""
    x_val, d_val = 0.5, 5
    fig = plot_time_series(model_fixture, x=x_val, d=d_val, t_range=(0, 10), title="Custom Time Series")
    ax = fig.axes[0]
    assert ax.get_title() == "Custom Time Series"
    assert ax.get_xlabel() == "Time (t)"
    assert ax.get_ylabel() == "Value"

    # Check for presence of lines for adaptability and coherence
    # Adaptability is the first line, Coherence is the second
    lines = ax.get_lines()
    assert len(lines) >= 2 # Should have at least adaptability and coherence lines, plus envelope
    labels = [line.get_label() for line in lines]
    assert "Adaptability A(x,d,t)" in labels
    assert "Coherence C(x,d,t)" in labels
    plt.close(fig)

    # Test default title
    fig_default_title = plot_time_series(model_fixture, x=x_val, d=d_val, t_range=(0, 10))
    ax_default = fig_default_title.axes[0]
    assert ax_default.get_title() == f"Time Series at x = {x_val}, d = {d_val}"
    plt.close(fig_default_title)

def test_plot_time_series_save(model_fixture):
    """Test that the time series plot can be saved to a file."""
    save_path = os.path.join(TEST_FIGURES_DIR, "time_series.png")
    if os.path.exists(save_path):
        os.remove(save_path)

    fig = plot_time_series(model_fixture, x=0.5, d=5, t_range=(0, 10), save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    plt.close(fig)
    # os.remove(save_path) # Clean up

from plotting import plot_power_spectrum

def test_plot_power_spectrum_runs(model_fixture):
    """Test that plot_power_spectrum runs without errors."""
    fig = plot_power_spectrum(model_fixture, x=0.5, d=5, t_range=(0, 10))
    assert fig is not None
    plt.close(fig)

def test_plot_power_spectrum_elements(model_fixture):
    """Test that the plot has a title, labels, and a power spectrum line."""
    x_val, d_val = 0.5, 5
    fig = plot_power_spectrum(model_fixture, x=x_val, d=d_val, t_range=(0, 10), title="Custom Power Spectrum")
    ax = fig.axes[0]
    assert ax.get_title() == "Custom Power Spectrum"
    assert ax.get_xlabel() == "Frequency (Hz)"
    assert ax.get_ylabel() == "Power Spectral Density"

    # Check for presence of the main power spectrum line
    lines = ax.get_lines()
    # The first line is usually the PSD line. Additional lines are theoretical frequencies.
    assert len(lines) > 0
    psd_line_labels = [line.get_label() for line in lines]
    assert "Power Spectrum" in psd_line_labels
    plt.close(fig)

    # Test default title
    fig_default_title = plot_power_spectrum(model_fixture, x=x_val, d=d_val, t_range=(0, 10))
    ax_default = fig_default_title.axes[0]
    assert ax_default.get_title() == f"Power Spectrum at x = {x_val}, d = {d_val}"
    plt.close(fig_default_title)

def test_plot_power_spectrum_save(model_fixture):
    """Test that the power spectrum plot can be saved to a file."""
    save_path = os.path.join(TEST_FIGURES_DIR, "power_spectrum.png")
    if os.path.exists(save_path):
        os.remove(save_path)

    fig = plot_power_spectrum(model_fixture, x=0.5, d=5, t_range=(0, 10), save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    plt.close(fig)
    # os.remove(save_path) # Clean up

from plotting import plot_exponential_decay

def test_plot_exponential_decay_runs(model_fixture):
    """Test that plot_exponential_decay runs without errors."""
    fig = plot_exponential_decay(model_fixture, x=0.25, d_range=(1, 20))
    assert fig is not None
    plt.close(fig)

def test_plot_exponential_decay_elements(model_fixture):
    """Test that the plot has correct suptitle and axis labels."""
    x_val = 0.25
    fig = plot_exponential_decay(model_fixture, x=x_val, d_range=(1, 20), title="Custom Exponential Decay")

    assert fig._suptitle.get_text() == "Custom Exponential Decay"

    # This plot creates two subplots
    assert len(fig.axes) == 2
    ax1, ax2 = fig.axes[0], fig.axes[1]

    assert ax1.get_xlabel() == "Depth (d)"
    assert ax1.get_ylabel() == "Adaptability A(x,d)"
    assert ax1.get_title() == "Linear Scale"

    assert ax2.get_xlabel() == "Depth (d)"
    assert ax2.get_ylabel() == "Adaptability A(x,d) (log scale)"
    assert ax2.get_title() == "Logarithmic Scale"
    plt.close(fig)

    # Test default suptitle
    fig_default_title = plot_exponential_decay(model_fixture, x=x_val, d_range=(1, 20))
    assert fig_default_title._suptitle.get_text() == f"Exponential Decay of Adaptability for x = {x_val}"
    plt.close(fig_default_title)

def test_plot_exponential_decay_save(model_fixture):
    """Test that the exponential decay plot can be saved to a file."""
    save_path = os.path.join(TEST_FIGURES_DIR, "exponential_decay.png")
    if os.path.exists(save_path):
        os.remove(save_path)

    fig = plot_exponential_decay(model_fixture, x=0.25, d_range=(1, 20), save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    plt.close(fig)
    # os.remove(save_path) # Clean up
import pandas as pd
from plotting import plot_modal_contributions

@pytest.fixture
def sample_modal_dataframe():
    """Provides a sample DataFrame for plot_modal_contributions tests."""
    data = {
        'Depth': np.linspace(1, 10, 5),
        'Mode 1': np.random.rand(5),
        'Mode 2': np.random.rand(5),
        'Normalized Entropy': np.random.rand(5)
    }
    df = pd.DataFrame(data)
    # Ensure mode columns are correctly identified (start with 'Mode')
    df['Mode 3'] = np.random.rand(5)
    return df

def test_plot_modal_contributions_runs(sample_modal_dataframe):
    """Test that plot_modal_contributions runs without errors."""
    fig = plot_modal_contributions(sample_modal_dataframe)
    assert fig is not None
    plt.close(fig)

def test_plot_modal_contributions_elements(sample_modal_dataframe):
    """Test that the plot has correct suptitle and axis labels."""
    fig = plot_modal_contributions(sample_modal_dataframe, title="Custom Modal Contributions")

    assert fig._suptitle.get_text() == "Custom Modal Contributions"

    # This plot creates two subplots
    assert len(fig.axes) == 2
    ax1, ax2 = fig.axes[0], fig.axes[1]

    assert ax1.get_xlabel() == "Depth (d)"
    assert ax1.get_ylabel() == "Relative Contribution"
    assert ax1.get_title() == "Modal Contributions vs Depth"

    assert ax2.get_xlabel() == "Depth (d)"
    assert ax2.get_ylabel() == "Normalized Entropy"
    assert ax2.get_title() == "Mode Distribution Complexity vs Depth"
    plt.close(fig)

    # Test default suptitle
    fig_default_title = plot_modal_contributions(sample_modal_dataframe)
    assert fig_default_title._suptitle.get_text() == "Modal Structure Analysis"
    plt.close(fig_default_title)

def test_plot_modal_contributions_save(sample_modal_dataframe):
    """Test that the modal contributions plot can be saved to a file."""
    save_path = os.path.join(TEST_FIGURES_DIR, "modal_contributions.png")
    if os.path.exists(save_path):
        os.remove(save_path)

    fig = plot_modal_contributions(sample_modal_dataframe, save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    plt.close(fig)
    # os.remove(save_path) # Clean up

from plotting import plot_multiple_landscapes

def test_plot_multiple_landscapes_runs(model_fixture):
    """Test that plot_multiple_landscapes runs without errors."""
    models = [model_fixture, AdaptabilityModel([1,3,5])]
    model_names = ["Model 1", "Model 2"]
    fig = plot_multiple_landscapes(models, model_names, x_range=(-1, 1), d_range=(1, 10))
    assert fig is not None
    plt.close(fig)

def test_plot_multiple_landscapes_elements(model_fixture):
    """Test that the plot has correct suptitle and individual plot titles."""
    models = [model_fixture, AdaptabilityModel([1,3,5])]
    model_names = ["N_ord=[1, 2, 3]", "N_ord=[1, 3, 5]"] # More descriptive for title checking

    fig = plot_multiple_landscapes(models, model_names, x_range=(-1, 1), d_range=(1, 10), title="Multiple Landscapes Test")

    assert fig._suptitle.get_text() == "Multiple Landscapes Test"

    # This plot creates a number of subplots equal to the number of models,
    # plus potentially a colorbar for each.
    # Assuming each landscape plot also creates a colorbar axis.
    assert len(fig.axes) == len(models) * 2

    # Check titles of subplots (axes)
    # The plot_adaptability_landscape function called internally sets the title of each ax
    # based on the model_name provided if not None, or a default.
    # Here, plot_multiple_landscapes passes model_names as titles to plot_adaptability_landscape.
    # We only check the first len(models) axes, assuming these are the main plot axes.
    main_plot_axes = fig.axes[:len(models)]
    for i, ax in enumerate(main_plot_axes):
        # plot_multiple_landscapes internally calls plot_adaptability_landscape,
        # which will use the passed model_names[i] as the title for each subplot.
        assert ax.get_title() == model_names[i]
        assert ax.get_xlabel() == "Configuration (x)" # From underlying plot_adaptability_landscape
        assert ax.get_ylabel() == "Depth (d)" # From underlying plot_adaptability_landscape
    plt.close(fig)

    # Test default suptitle (it's None by default for this function if not provided)
    fig_default_title = plot_multiple_landscapes(models, model_names, x_range=(-1, 1), d_range=(1, 10))
    assert fig_default_title._suptitle is None # Default behavior
    plt.close(fig_default_title)

def test_plot_multiple_landscapes_save(model_fixture):
    """Test that the multiple landscapes plot can be saved to a file."""
    save_path = os.path.join(TEST_FIGURES_DIR, "multiple_landscapes.png")
    if os.path.exists(save_path):
        os.remove(save_path)

    models = [model_fixture, AdaptabilityModel([1,3,5])]
    model_names = ["Model 1", "Model 2"]
    fig = plot_multiple_landscapes(models, model_names, x_range=(-1, 1), d_range=(1, 10), save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    plt.close(fig)
    # os.remove(save_path) # Clean up

from plotting import plot_spectral_fingerprints

def test_plot_spectral_fingerprints_runs(model_fixture):
    """Test that plot_spectral_fingerprints runs without errors."""
    models = [model_fixture, AdaptabilityModel([1,3,5])]
    model_names = ["Model 1", "Model 2"]
    fig = plot_spectral_fingerprints(models, model_names, x=0.5, d=5, t_range=(0,10))
    assert fig is not None
    plt.close(fig)

def test_plot_spectral_fingerprints_elements(model_fixture):
    """Test that the plot has correct suptitle and individual plot titles."""
    models = [model_fixture, AdaptabilityModel([1,3,5])]
    model_names = ["Harmonic", "Odd Harmonic"] # For checking titles
    x_val, d_val = 0.5, 5

    fig = plot_spectral_fingerprints(models, model_names, x=x_val, d=d_val, t_range=(0,10), title="Spectral Fingerprints Test")

    assert fig._suptitle.get_text() == "Spectral Fingerprints Test"

    # This plot creates a subplot for each model
    assert len(fig.axes) == len(models)

    for i, ax in enumerate(fig.axes):
        # The title for each subplot is f'{model_names[i]}: N_ord = {models[i].n_ord}'
        expected_subplot_title = f'{model_names[i]}: N_ord = {models[i].n_ord}'
        assert ax.get_title() == expected_subplot_title
        assert ax.get_xlabel() == "Frequency (Hz)"
        assert ax.get_ylabel() == "Power Spectral Density"
    plt.close(fig)

    # Test default suptitle
    fig_default_title = plot_spectral_fingerprints(models, model_names, x=x_val, d=d_val, t_range=(0,10))
    expected_default_suptitle = f'Spectral Fingerprints at x = {x_val}, d = {d_val}'
    assert fig_default_title._suptitle.get_text() == expected_default_suptitle
    plt.close(fig_default_title)

def test_plot_spectral_fingerprints_save(model_fixture):
    """Test that the spectral fingerprints plot can be saved to a file."""
    save_path = os.path.join(TEST_FIGURES_DIR, "spectral_fingerprints.png")
    if os.path.exists(save_path):
        os.remove(save_path)

    models = [model_fixture, AdaptabilityModel([1,3,5])]
    model_names = ["Model 1", "Model 2"]
    fig = plot_spectral_fingerprints(models, model_names, x=0.5, d=5, t_range=(0,10), save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    plt.close(fig)
    # os.remove(save_path) # Clean up

def test_generic_matplotlib_plot_save():
    """Tests that a generic matplotlib plot can be created and saved."""
    save_path = os.path.join(TEST_FIGURES_DIR, "generic_test_plot.png")
    if os.path.exists(save_path):
        os.remove(save_path)

    x_coords = np.linspace(0, 10, 100)
    y_coords = np.sin(x_coords)

    fig = plt.figure()
    plt.plot(x_coords, y_coords)
    plt.title("Generic Test Plot")
    plt.savefig(save_path)
    plt.close(fig)

    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0
    # os.remove(save_path) # Clean up
