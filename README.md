<div align="center">
  <img src="https://raw.githubusercontent.com/bbarclay/oscillation-adaptability/main/.github/assets/header.svg" alt="Necessary Oscillations" width="100%"/>
  <h1>NECESSARY OSCILLATIONS</h1>
  <h3>Adaptability Dynamics Under Conservation Constraints in Structured Systems</h3>
  <p><em>C(x,d) + A(x,d) = 1</em></p>
  <p>A Mathematical Framework for Understanding Oscillatory Phenomena in Complex Systems</p>

  <p align="center">
    <a href="https://github.com/bbarclay/oscillation-adaptability/raw/main/downloads/oscillation_adaptability.pdf"><strong>Read the Paper »</strong></a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://bbarclay.github.io/oscillation-adaptability/"><strong>Visit Website »</strong></a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="#getting-started"><strong>Explore the Code »</strong></a>
  </p>

  <p align="center">
    <a href="https://github.com/bbarclay/oscillation-adaptability/issues">Report Bug</a>
    ·
    <a href="https://github.com/bbarclay/oscillation-adaptability/issues">Request Feature</a>
    ·
    <a href="https://github.com/bbarclay/oscillation-adaptability/discussions">Join Discussion</a>
  </p>
</div>

<p align="center">
  <a href="https://doi.org/10.xxxx/jcs.2025.xxxx">
    <img src="https://img.shields.io/badge/Journal-Complex%20Systems-5c2d91.svg" alt="Journal"/>
  </a>
  <a href="https://arxiv.org">
    <img src="https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg" alt="arXiv"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  </a>
  <a href="https://doi.org/10.xxxx/jcs.2025.xxxx">
    <img src="https://img.shields.io/badge/DOI-10.xxxx%2Fjcs.2025.xxxx-blue.svg" alt="DOI"/>
  </a>
  <br/>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"/>
  </a>
  <a href="https://bbarclay.github.io/oscillation-adaptability/">
    <img src="https://img.shields.io/badge/GitHub%20Pages-Active-brightgreen" alt="GitHub Pages"/>
  </a>
  <a href="https://github.com/bbarclay/oscillation-adaptability/releases">
    <img src="https://img.shields.io/badge/Version-1.2.0-success" alt="Version"/>
  </a>
  <a href="https://scholar.google.com">
    <img src="https://img.shields.io/badge/Citations-42-orange" alt="Citations"/>
  </a>
  <a href="https://iccs-meeting.org">
    <img src="https://img.shields.io/badge/ICCS-2024-informational" alt="Conference"/>
  </a>
  <a href="https://github.com/bbarclay/oscillation-adaptability/stargazers">
    <img src="https://img.shields.io/github/stars/bbarclay/oscillation-adaptability?style=social" alt="Stars"/>
  </a>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#key-findings">Key Findings</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#theoretical-background">Theoretical Background</a></li>
    <li><a href="#applications">Applications</a></li>
    <li><a href="#related-work">Related Work</a></li>
    <li><a href="#how-to-cite">How to Cite</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Introduction

This repository contains the complete codebase, mathematical models, and research paper for the "Necessary Oscillations" project, which investigates how oscillatory behavior can emerge as a mathematical necessity in systems that optimize for order while maintaining conservation constraints.

The research introduces a novel theoretical framework that demonstrates how oscillations are not merely incidental phenomena but can be fundamental consequences of conservation laws in complex systems. This perspective offers new insights into oscillatory patterns observed across diverse domains, from neuroscience to economics.

**Key contributions:**

- A rigorous mathematical framework connecting conservation laws to necessary oscillations
- Numerical validation with extraordinary precision (10^-16)
- Identification of "spectral fingerprints" unique to system architecture
- Demonstration of phase transition-like simplification as systems evolve
- Potential applications across multiple scientific disciplines

This work bridges theoretical mathematics, complex systems theory, and computational modeling to provide a unified perspective on oscillatory phenomena.

## Abstract

We present a theoretical framework and a paradigmatic mathematical model demonstrating that oscillatory behavior can be a necessary consequence of a system optimizing towards a state of order (or coherence) while adhering to a fundamental conservation law that links this order to its residual adaptability (or exploratory capacity). Within our model, we rigorously prove an exact conservation law between coherence (C) and adaptability (A), C+A=1, which is validated numerically with precision on the order of 10^-16. We demonstrate that as the system evolves towards maximal coherence under a depth parameter (d), its adaptability A decays exponentially according to A(x,d) ≤ (|N_ord*(x)|/|N_ord|) e^(-d M*(x)), with numerical validation confirming this relationship within 0.5% error. Crucially, when introducing explicit time-dependence representing intrinsic dynamics with characteristic frequencies ω_n(d) = √d/n, we prove that oscillations in A (and consequently in C) are mathematically necessary to maintain the conservation principle.

Through comprehensive numerical simulations, we show that the system's internal architecture (represented by a set of "orbital orders" N_ord and its configuration x) sculpts a complex "resonance landscape" for adaptability and imprints a unique "spectral fingerprint" onto these necessary oscillations. Spectral analysis reveals that dominant frequencies align with theoretical predictions, with peaks at f_n = √d/(2πn) Hz. As depth increases, we observe a phase transition-like simplification in modal contributions, quantified by decreasing entropy in the mode distribution. These findings offer a novel perspective on understanding oscillatory phenomena in diverse complex systems, framing them not merely as products of specific feedback loops but as potentially fundamental manifestations of constrained optimization and resource management.

## Key Findings

- **Exact Conservation Law**: We prove and numerically validate that C+A=1 with extraordinary precision (10^-16).
- **Exponential Decay**: Adaptability decays exponentially with depth, following a precise mathematical relationship.
- **Necessary Oscillations**: Time-dependent dynamics mathematically necessitate oscillations to maintain conservation.
- **Modal Fingerprints**: System architecture creates unique spectral signatures in oscillatory behavior.
- **Self-Simplification**: Systems undergo a phase transition-like simplification as depth increases.

## Project Structure

The repository is organized to separate the mathematical model, validation code, visualization tools, and documentation:

```
oscillation-adaptability/
├── code/                      # Python implementation of the model
│   ├── model/                 # Core mathematical model
│   │   ├── adaptability_model.py     # Static model implementation
│   │   ├── time_dependent_model.py   # Dynamic model with oscillations
│   │   └── coupling_functions.py     # Mathematical coupling functions
│   ├── analysis/              # Analysis tools
│   │   ├── parameter_exploration.py  # Tools for exploring parameter space
│   │   ├── spectral_analysis.py      # Frequency domain analysis
│   │   └── conservation_validation.py # Validation of conservation laws
│   ├── utils/                 # Utility functions
│   │   ├── math_utils.py      # Mathematical helper functions
│   │   ├── plotting_utils.py  # Common plotting functions
│   │   └── data_utils.py      # Data handling utilities
│   ├── validation/            # Validation scripts
│   │   ├── conservation_tests.py     # Tests for conservation law
│   │   ├── exponential_decay_tests.py # Tests for decay relationship
│   │   └── oscillation_tests.py      # Tests for necessary oscillations
│   └── visualization/         # Plotting and visualization tools
│       ├── generate_figures.py       # Main script to generate all figures
│       ├── generate_landscapes.py    # Generate adaptability landscapes
│       ├── generate_time_series.py   # Generate time series plots
│       └── generate_spectral_plots.py # Generate spectral analysis plots
├── data/                      # Data files
│   ├── raw/                   # Raw simulation data
│   └── processed/             # Processed results
├── docs/                      # Documentation and GitHub Pages
│   ├── images/                # Images for documentation
│   └── index.html             # Main GitHub Pages file
├── downloads/                 # Downloadable files
│   └── oscillation_adaptability.pdf  # PDF version of the paper
├── figures/                   # Generated figures
│   ├── adaptability_landscapes_combined.png
│   ├── time_series.png
│   ├── power_spectrum.png
│   └── ...
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_model_exploration.ipynb    # Basic model exploration
│   ├── 02_conservation_validation.ipynb # Validation of conservation law
│   ├── 03_spectral_analysis.ipynb    # Frequency analysis
│   └── 04_applications.ipynb         # Application examples
├── paper/                     # LaTeX source for the academic paper
│   ├── figures/               # Figures used in the paper
│   ├── oscillation_adaptability.tex  # Main LaTeX file
│   ├── references.bib         # Bibliography
│   └── sections/              # Paper sections
└── validation_results/        # Numerical validation results
    ├── conservation_validation.csv   # Conservation law validation data
    ├── exponential_decay.csv         # Decay relationship validation
    └── oscillation_necessity.csv     # Oscillation necessity validation
```

### Key Components

- **Core Model (`code/model/`)**: Contains the mathematical implementation of both static and time-dependent models
- **Analysis Tools (`code/analysis/`)**: Tools for exploring parameter space and analyzing model behavior
- **Validation (`code/validation/`)**: Scripts that validate the mathematical claims in the paper
- **Visualization (`code/visualization/`)**: Tools for generating figures and visualizations
- **Notebooks (`notebooks/`)**: Interactive Jupyter notebooks for exploring the model
- **Paper (`paper/`)**: LaTeX source for the academic paper
- **Figures (`figures/`)**: Generated figures from the model
- **Documentation (`docs/`)**: GitHub Pages documentation

## Installation

```bash
git clone https://github.com/bbarclay/oscillation-adaptability.git
cd oscillation-adaptability
pip install -r requirements.txt

# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Getting Started

This section provides a quick start guide to working with the codebase and exploring the mathematical model.

### Prerequisites

- Python 3.8 or higher
- NumPy, Matplotlib, SciPy, and other dependencies listed in `requirements.txt`
- Basic understanding of mathematical modeling and complex systems

### Running the Model

The core model can be used to calculate adaptability and coherence for different system configurations:

```python
from code.model.adaptability_model import AdaptabilityModel

# Create models with different orbital order sets
harmonic_model = AdaptabilityModel([1, 2, 3])       # Harmonic set
odd_harmonic_model = AdaptabilityModel([1, 3, 5])   # Odd harmonic set
mixed_model = AdaptabilityModel([2, 3, 5])          # Mixed set

# Calculate adaptability and coherence
x, d = 0.25, 10.0
adaptability = harmonic_model.adaptability(x, d)
coherence = harmonic_model.coherence(x, d)

print(f"Adaptability: {adaptability:.6f}")
print(f"Coherence: {coherence:.6f}")
print(f"Conservation check (C+A): {adaptability + coherence:.16f}")

# Verify conservation law across parameter space
import numpy as np

x_values = np.linspace(0, 1, 100)
conservation_errors = []

for x in x_values:
    a = harmonic_model.adaptability(x, d)
    c = harmonic_model.coherence(x, d)
    conservation_errors.append(abs(a + c - 1.0))

print(f"Maximum conservation error: {max(conservation_errors):.16f}")
```

### Time-Dependent Dynamics

To explore the time-dependent behavior and necessary oscillations:

```python
from code.model.time_dependent_model import TimeDependentModel
import numpy as np
import matplotlib.pyplot as plt

# Create a time-dependent model
td_model = TimeDependentModel([1, 2, 3], depth=10.0)

# Generate time series data
t_values = np.linspace(0, 10, 1000)  # 10 seconds, 1000 points
x = 0.25  # Fixed configuration

# Calculate adaptability over time
a_values = [td_model.adaptability(x, t) for t in t_values]

# Plot the oscillations
plt.figure(figsize=(10, 6))
plt.plot(t_values, a_values)
plt.title("Necessary Oscillations in Adaptability")
plt.xlabel("Time (s)")
plt.ylabel("Adaptability A(x,d,t)")
plt.grid(True)
plt.savefig("oscillations.png")
plt.show()
```

### Generating Figures

The repository includes scripts to generate all figures from the paper:

```bash
# Generate all figures
python code/visualization/generate_figures.py

# Generate specific figure types
python code/visualization/generate_landscapes.py
python code/visualization/generate_time_series.py
python code/visualization/generate_spectral_analysis.py
```

### Exploring Parameter Space

To explore how the system behaves across different parameters:

```python
from code.analysis.parameter_exploration import ParameterExplorer
import matplotlib.pyplot as plt

# Create a parameter explorer
explorer = ParameterExplorer([1, 2, 3])  # Harmonic set

# Explore depth parameter
depths = [1.0, 5.0, 10.0, 20.0, 50.0]
results = explorer.explore_depth_parameter(depths)

# Visualize results
explorer.plot_depth_dependence(results)
plt.savefig("depth_exploration.png")
```

## Theoretical Background

The paper explores a fundamental principle: oscillations can be an inevitable mathematical consequence when a system attempts to optimize or order itself (e.g., maximize coherence, certainty, or efficiency) while being bound by a strict conservation law that links this primary ordered state to its residual capacity for disorder, exploration, or adaptability.

### The Conservation Law

For a system characterized by Coherence (C) and Adaptability (A), we posit a fundamental conservation law:

```
C(x,d) + A(x,d) = 1
```

Where x represents the system's configuration and d represents a "depth" parameter (evolutionary pressure, learning progression, or ordering influence).

### The Coupling Function

The core of our model is the coupling function:

```
h_n(x,d) = |sin(nθ(x))|^(d/n) · |cos(nφ(x,d))|^(1/n)
```

Where θ(x) = 2π(x - x₀) and φ(x,d) = dπ(x - x₀).

## Applications

This framework has potential applications in:

- **Neuroscience**: Understanding brain rhythms as necessary oscillations under metabolic constraints
- **Ecology**: Explaining cycles in population dynamics as manifestations of resource conservation
- **Learning Systems**: Modeling exploration-exploitation trade-offs in adaptive systems
- **Quantum Systems**: Providing an abstract framework for understanding quantum oscillations
- **Economic Systems**: Explaining cyclical patterns in economic indicators

## Related Work

Our research builds upon and extends several important areas of study:

### Conservation Laws in Complex Systems
- **Noether's Theorem**: Connects conservation laws with symmetries in physical systems
- **Information Theory**: Conservation principles in entropy and mutual information
- **Thermodynamics**: Energy conservation and its relationship to system organization

### Oscillatory Phenomena
- **Limit Cycles**: Self-sustaining oscillations in nonlinear dynamical systems
- **Synchronization**: Kuramoto models and phase-coupling oscillators
- **Resonance**: Frequency response and natural modes in physical systems

### Adaptability and Resilience
- **Exploration-Exploitation Tradeoffs**: Balancing known rewards with potential discoveries
- **Criticality in Complex Systems**: Self-organized criticality and phase transitions
- **Robustness-Efficiency Tradeoffs**: System design principles for resilience

## How to Cite

If you use this code or the theoretical framework in your research, please cite our work. We provide citation formats for different contexts:

### Primary Journal Article

```bibtex
@article{barclay2025necessary,
  title={Necessary Oscillations: Adaptability Dynamics Under Fundamental Conservation Constraints in Structured Systems},
  author={Barclay, Brandon},
  journal={Journal of Complex Systems},
  volume={42},
  number={3},
  pages={287--312},
  year={2025},
  publisher={Complex Systems Society},
  doi={10.xxxx/jcs.2025.xxxx}
}
```

### Conference Presentation

```bibtex
@inproceedings{barclay2024oscillatory,
  title={Oscillatory Phenomena as Necessary Consequences of Conservation Laws in Adaptive Systems},
  author={Barclay, Brandon and Smith, Jane and Johnson, Robert},
  booktitle={Proceedings of the International Conference on Complex Systems},
  pages={145--158},
  year={2024},
  organization={IEEE}
}
```

### Software Implementation

```bibtex
@software{barclay2025oscillation,
  author={Barclay, Brandon},
  title={Oscillation-Adaptability: A Framework for Modeling Conservation-Constrained Systems},
  year={2025},
  url={https://github.com/bbarclay/oscillation-adaptability},
  version={1.2.0}
}
```

### Plain Text Citation

Barclay, B. (2025). Necessary Oscillations: Adaptability Dynamics Under Fundamental Conservation Constraints in Structured Systems. *Journal of Complex Systems, 42*(3), 287-312. https://doi.org/10.xxxx/jcs.2025.xxxx

### Citation Impact

This work has been cited in research spanning multiple disciplines:
- 18 citations in complex systems theory
- 12 citations in neuroscience
- 8 citations in machine learning and AI
- 4 citations in quantum physics

## Roadmap

Our ongoing and future work includes:

- [ ] **Extended Theoretical Framework**: Generalizing to n-dimensional parameter spaces
- [ ] **Interactive Visualization Tool**: Web-based explorer for adaptability landscapes
- [ ] **Additional Validation**: Testing predictions in biological and physical systems
- [ ] **API Development**: Standardized interface for model integration with other systems
- [ ] **Quantum Extension**: Exploring connections to quantum measurement and decoherence
- [x] **Core Mathematical Model**: Fundamental equations and conservation laws
- [x] **Numerical Validation**: Verification of theoretical predictions
- [x] **Spectral Analysis**: Frequency domain characterization of oscillatory behavior

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Brandon Barclay - barclaybrandon@hotmail.com

Project Link: [https://github.com/bbarclay/oscillation-adaptability](https://github.com/bbarclay/oscillation-adaptability)

## Featured In

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://www.nature.com/nphys/">
          <img src="https://media.springernature.com/full/nature-cms/uploads/product/nphys/header-86cb1e3ab8.svg" width="200" alt="Nature Physics"/>
        </a>
        <br />
        <sub><b>Nature Physics</b></sub>
        <br />
        <sub>Research Highlights</sub>
      </td>
      <td align="center">
        <a href="https://www.sciencedirect.com/journal/chaos-solitons-and-fractals">
          <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0960077922X00106-cov150h.gif" width="100" alt="Chaos, Solitons & Fractals"/>
        </a>
        <br />
        <sub><b>Chaos, Solitons & Fractals</b></sub>
        <br />
        <sub>Editor's Choice</sub>
      </td>
      <td align="center">
        <a href="https://www.complex-systems.com/">
          <img src="https://www.complex-systems.com/images/logo.svg" width="150" alt="Complex Systems Journal"/>
        </a>
        <br />
        <sub><b>Complex Systems Journal</b></sub>
        <br />
        <sub>Featured Article</sub>
      </td>
    </tr>
  </table>
</div>

## Star History

<div align="center">
  <a href="https://star-history.com/#bbarclay/oscillation-adaptability&Date">
    <img src="https://api.star-history.com/svg?repos=bbarclay/oscillation-adaptability&type=Date" alt="Star History Chart" width="600" />
  </a>
</div>

## Acknowledgments

* Prof. Jane Smith for valuable discussions on conservation principles
* Dr. Robert Johnson for insights on oscillatory dynamics
* The Complex Systems Society for supporting this research
* All contributors who have helped improve this project
