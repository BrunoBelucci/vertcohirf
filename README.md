# VertCoHiRF: Decentralized Vertical Clustering Beyond k-means via Heterogeneous Structural Consensus

This is the supplementary code for the paper VertCoHiRF: Decentralized Vertical Clustering Beyond k-means via Heterogeneous Structural Consensus

## Installing the requirements

To install the requirements, run the following command:

```bash
conda create --name vertcohirf --file package-list.txt
```

We have also a supplementary dependency on the `cohirf` library. To install, download it from
https://github.com/BrunoBelucci/cohirf and follow the installation instructions. 

## Running the experiments

### Multi-modal experiment

The experiment can be run through the jupyter notebook in `notebooks/custom.ipynb` or the equivalent python script
 `notebooks/custom.py`.

### Byzantine attack experiment

The experiment can be run through the jupyter notebook in `notebooks/attack_rank.ipynb` or the equivalent python script
 `notebooks/attack_rank.py`.

### Real World Datasets experiment

We have provided the commands to run all the experiments in the bash scripts in `scripts/real-*.sh`. Even though we provide the code to run everything at once for each experiment, we recommend running each combination of model and dataset separately.