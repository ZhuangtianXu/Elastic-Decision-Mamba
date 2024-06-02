# Elastic Decision Mamba


[Elastic Decision Mamba](https://github.com/ZhuangtianXu/Elastic-Decision-Mamba/), Zhuangtian Xu.

Elastic Decision Mamba is a novel offline RL approach that incorporating Mamba architecture into EDT.


## Bibtex

```
@article{xu2024elastic,
  title={Elastic Decision Mamba},
  author={Xu, Zhuangtian},
  journal={arXiv preprint arXiv:2307.02484},
  year={2024}
}
```


## Installation
We prepared a Dockerfile and bash scripts to set up the environment.

1. Build the Docker image and start a Docker container 
```bash
# Download the code from this repo
git clone https://github.com/ZhuangtianXu/Elastic-Decision-Mamba.git
cd Elastic-Decision-Mamba
bash build_image.sh
bash start_container.sh
```

## Training
1. Download D4RL datasets
```bash
cd /workspace
python data/download_d4rl_datasets.py
```

2. Train the EDT agent
```bash
python scripts/train_edm.py --env hopper --dataset medium-replay
```

## Evaluation
```bash
python scripts/eval_edm.py --chk_pt_name saved_model_name_from_training.pt
```

## Acknowledgement
The implementation of EDM is based on [Elastic-DT](https://github.com/kristery/Elastic-DT) and [Mamba](https://github.com/state-spaces/mamba)

