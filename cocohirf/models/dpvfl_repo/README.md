# Differentially Private Vertical Federated Clustering

### Example command

Running command samples:
* sample run `DPLSF + DPFMPS-2PEst`: ` python3 ./main.py --solution V2way --config ./configs/mg/lsh_fmsketch.yaml
`
* sample run `DPLSF + DPFMPS-BasicEst`: ` python3 ./main.py --solution VPC --config ./configs/mg/lsh_fmsketch.yaml
`
* sample run `DPLSF + IND-LAP`: ` python3 ./main.py --solution VPC --config ./configs/mg/lsh_ind_lsp.yaml
`
* sample run `DPLSF + LDP-AGG-2PEst`: ` python3 ./main.py --solution V2way --config ./configs/mg/lsh_ldp.yaml
`


### Dataset links:

* Loan dataset: https://www.kaggle.com/c/home-credit-default-risk/data?select=application_train.csv
* New York taxi: https://www.kaggle.com/dansbecker/new-york-city-taxi-fare-prediction
* Letter dataset: https://archive.ics.uci.edu/ml/datasets/letter+recognition 