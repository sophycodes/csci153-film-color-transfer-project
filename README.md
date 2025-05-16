# Color Transfer Between Movie Scenes

**Adapting Cinematic Color Palettes Across Film Genres**

## Overview

This project explores automated color transfer between film scenes using a cluster-based framework and five palette matching strategies. The objective is to transform the color style of one scene (e.g., *10 Things I Hate About You*) to resemble that of another (e.g., *The Matrix*, *La La Land*, *Harry Potter*) while preserving scene semantics.

Implemented in Python and tested on the Harvey Mudd College HPC cluster.

## Features

* Five color transfer algorithms:

  * Relative distance-based mapping
  * Target frequency (many-to-one and one-to-one)
  * Source frequency (one-to-one)
  * Depth-based transfer using monocular depth estimation
* Preprocessing with CIELAB color space and brightness filtering
* Quantitative evaluation using RGB and HSV histograms with Bhattacharyya distance
* Visual comparison for qualitative assessment

## Repository Structure

```
color_transfer_files/
├── imgs/                        # Input target frames
├── kaggle/                      # Source movie reference frames
├── __pycache__/
├── our_functions.py             # Core logic
├── our_test_functions.py        # Test utilities
├── Final_Color_Transfer_Test_Results.ipynb  # Notebook demo (optional)
├── requirements.txt             # Dependencies
└── .gitignore
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/sophycodes/csci153-film-color-transfer-project.git
cd csci153-film-color-transfer-project
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the color transfer methods:

```bash
python our_functions.py
```

4. Optionally, open the notebook:

```bash
jupyter notebook Final_Color_Transfer_Test_Results.ipynb
```

## Results

* Best statistical performance: Method 2 (Target Frequency Many-to-One)
* Best visual results: Method 4 (Source Frequency One-to-One)
* Quantitative evaluation via Bhattacharyya distance confirms method effectiveness
* Histogram visualizations included for each transfer method

## Authors

* Sophy Figaroa, Pomona College
* Theodore Julien, Harvey Mudd College
