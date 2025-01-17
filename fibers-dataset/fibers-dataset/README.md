# Fibers Dataset

This project generates a fibers dataset from skeletons for volumetric annotation of Herculaneum papyri.

## Installation

To set up the environment, run the following commands:

```bash
conda env create -f environment.yml
conda activate fibers-dataset
vesuvius.accept_terms --yes
```

## Example Usage
```bash
python fibers-dataset-generator.py --nml_path fibers_s5_06500z_02000y_04000x_v03.nml --size 500 --output_folder output
```

The example annotation was done by Elian Rafael Dal Prá


