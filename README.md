# Code of paper Challenges and opportunities in applying Neural Point Processes to large scale industry data
We present in this repository code to train, evaluate Neural Temporal Point Processes (TPPs), as described in our work: "Challenges and opportunities in applying Neural Point Processes to large scale industry data"

## Contact
* Dominykas Šeputis dom.seputis@gmail.com

## Data
We use three datasets:
* Simulated Hawkes process
* Stack Overflow users' activity
* Vinted platform members' actions. If needed to access Vinted data, request for the data via email provided in the contact section

## Setup instructions
1. We use [poetry](https://python-poetry.org) as python dependency manager. To setup python virtual environment, first, follow [the instructions](https://python-poetry.org/docs/) to install poetry on your machine.
2. Run `$ poetry install` to setup virtual environment
3. Run `$ source ./venv/bin/activate` to activate the newly initiated virtual environment
4. Authenticate with [wandb](https://docs.wandb.ai/quickstart) tool to track experiments
5. To prepare datasets for training and evaluation steps, process them by running `$ sh ./runs/data_preparation.sh`
6. To replicate experiments, run one of the experiments' scripts inside `./runs/`.
7. Alternatively, run specific experiment by running `$ python -m scripts.train --experiment <EXPERIMENT_NAME> --model-name <MODEL_NAME> --split-num <SPLITS_COUNT>`

## Structure of the repository
```
├── config <- Config files used for data processing and experimentation
│   ├── data
│   └── experiments <- Subdirectories of different experiments based on the dataset
│       ├── hawkes
│       ├── stack_overflow
│       └── vinted
├── data <- Place where raw and processed/generated data is stored
│   └── raw
│       └── stack_overflow
├── runs <- .sh files that run multiple python scripts
├── scripts <- Place where training and data processing python scripts are held
│   └── data
└── src <- Source files
    ├── datasets <- Datasets' implementations
    ├── models <- Models' implementations
    └── utils <- Various utility functions
```
