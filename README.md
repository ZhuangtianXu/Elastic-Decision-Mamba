# Elastic Decision Mamba

[[Project Page]](https://kristery.github.io/edt/) [[Paper]](https://arxiv.org/abs/2307.02484)
-----

[Elastic Decision Transformer](https://kristery.github.io/edt/), Yueh-Hua Wu, Xiaolong Wang, Masashi Hamaya, NeurIPS 2023.

Elastic Decision Transformer is a novel Decision Transformer approach that enables trajectory stitching by adopting different history length. Elastic Decision Mamba replaces the transformer in EDT with the currently popular Mamba state space model architecture.


## Bibtex

```
@article{wu2023elastic,
  title={Elastic Decision Transformer},
  author={Wu, Yueh-Hua and Wang, Xiaolong and Hamaya, Masashi},
  journal={arXiv preprint arXiv:2307.02484},
  year={2023}
}
```


## Installation
We prepared a Dockerfile and bash scripts to set up the environment.

1. Build the Docker image and start a Docker container 
```bash
# Download the code from this repo
git clone https://github.com/kristery/Elastic-DT.git
cd Elastic-DT
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

