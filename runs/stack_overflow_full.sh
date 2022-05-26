for split_num in 1 2 3 4 5
do
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_lnm.yaml python -m scripts.train --experiment stack_overflow --model-name sa_lnm --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_rmtpp_poisson.yaml python -m scripts.train --experiment stack_overflow --model-name gru_rmtpp_poisson --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/diffeq.yaml python -m scripts.train --experiment stack_overflow --model-name diffeq --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_mlp_cm_tanh.yaml python -m scripts.train --experiment stack_overflow --model-name gru_mlp_cm_tanh --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_cond_poisson.yaml python -m scripts.train --experiment stack_overflow --model-name sa_cond_poisson --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_mlp_mc.yaml python -m scripts.train --experiment stack_overflow --model-name gru_mlp_mc --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_mlp_cm.yaml python -m scripts.train --experiment stack_overflow --model-name gru_mlp_cm --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_sa_mc.yaml python -m scripts.train --experiment stack_overflow --model-name sa_sa_mc --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_mlp_mc.yaml python -m scripts.train --experiment stack_overflow --model-name sa_mlp_mc --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_mlp_cm.yaml python -m scripts.train --experiment stack_overflow --model-name sa_mlp_cm --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_rmtpp.yaml python -m scripts.train --experiment stack_overflow --model-name gru_rmtpp --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_rmtpp_poisson.yaml python -m scripts.train --experiment stack_overflow --model-name sa_rmtpp_poisson --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_rmtpp.yaml python -m scripts.train --experiment stack_overflow --model-name sa_rmtpp --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_sa_cm.yaml python -m scripts.train --experiment stack_overflow --model-name gru_sa_cm --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/gru_sa_mc_softmax.yaml python -m scripts.train --experiment stack_overflow --model-name gru_sa_mc_softmax --device cuda --split-num $split_num
    dvc run -n train -f -d scripts/train.py -d config/experiments/stack_overflow/sa_sa_cm.yaml python -m scripts.train --experiment stack_overflow --model-name sa_sa_cm --device cuda --split-num $split_num
done
echo "All experiments were completed!"
