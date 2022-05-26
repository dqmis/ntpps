for split_num in 1 2 3 4 5
do
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_lnm.yaml python -m scripts.train --experiment stack_overflow --model-name sa_lnm --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_cond_poisson.yaml python -m scripts.train --experiment stack_overflow --model-name sa_cond_poisson --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_mlp_mc.yaml python -m scripts.train --experiment stack_overflow --model-name gru_mlp_mc --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_sa_mc.yaml python -m scripts.train --experiment stack_overflow --model-name sa_sa_mc --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_mlp_mc.yaml python -m scripts.train --experiment stack_overflow --model-name sa_mlp_mc --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_rmtpp_poisson.yaml python -m scripts.train --experiment stack_overflow --model-name sa_rmtpp_poisson --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_sa_mc_softmax.yaml python -m scripts.train --experiment stack_overflow --model-name gru_sa_mc_softmax --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/diffeq.yaml python -m scripts.train --experiment stack_overflow --model-name diffeq --split-num $split_num
done
echo "All experiments were completed!"
