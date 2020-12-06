### This project explore the catastrophic forgetting in GNN models.
# run like this:

python train.py --dataset 'corafull' \
--basemodel 'GAT' \
--method 'twp' \
--lambda_l 10000 \
--lambda_t 10000 \
--beta 0.01 \
--gpu 6 \
--seed 1 \
--n-tasks 9