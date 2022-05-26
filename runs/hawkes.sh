for split_num in 1 2 3 4 5
do
    dvc run -n train -f -d scripts/train.py -d config/experiments/hawkes/sa_lnm.yaml python -m scripts.train --experiment hawkes --model-name sa_lnm --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/hawkes/sa_cond_poisson.yaml python -m scripts.train --experiment hawkes --model-name sa_cond_poisson --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/hawkes/gru_mlp_mc.yaml python -m scripts.train --experiment hawkes --model-name gru_mlp_mc --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/hawkes/sa_sa_mc.yaml python -m scripts.train --experiment hawkes --model-name sa_sa_mc --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/hawkes/sa_mlp_mc.yaml python -m scripts.train --experiment hawkes --model-name sa_mlp_mc --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/hawkes/sa_rmtpp_poisson.yaml python -m scripts.train --experiment hawkes --model-name sa_rmtpp_poisson --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/hawkes/gru_sa_mc_softmax.yaml python -m scripts.train --experiment hawkes --model-name gru_sa_mc_softmax --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/hawkes/diffeq.yaml python -m scripts.train --experiment hawkes --model-name diffeq --split-num $split_num
done
echo "All experiments were completed!"
