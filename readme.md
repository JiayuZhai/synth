# Project Title

Generative Models for Generating Synthetic Social Data

## Getting Started

The package includes these directories.
data: raw data and pre-processed data. Some synthetic data will be generated here.
model: trained model.
figure: figures generated for the report.

And these python scripts.
clean.py: script for data cleaning.
analysis.py: script for isolated PCA analysis.
experiment.py: script for GAN model training.
visual.py: script for generating synthesis data and evaluations.
model.py: all training models and evaluation models.


### Prerequisites

The package needs numpy, matplotlib, pytorch, sklearn and scipy.

```
conda install numpy
conda install scipy
conda install scikit-learn
sudo apt-get install python3-matplotlib
conda install pytorch-cpu torchvision-cpu -c pytorch
```

## Running the experiment

Clean raw data first.
```
python clean.py
```

Use the script file exp.sh to start experiment. It will train three GAN model and generate their synthetic data. Then it synthesis data by another two methods. Finally, we compared all method by their evaluation measures.

```
bash exp.sh
```
