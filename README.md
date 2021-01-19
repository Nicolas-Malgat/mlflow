<p  align="center">
<h1  align="center">MLflow</h3>
</p>

## Table of Contents

*  [Getting Started](#getting-started)

*  [Installation](#installation)

*  [Usage](#usage)

## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo
```git
git clone https://github.com/Nicolas-Malgat/mlflow.git
```
2. Create a conda virtual environment with
```bash
conda create --name <env> --file conda_genere_par_conda.yml
```
## Usage

#### To generate a new MLflow run
- Run [entrainement.ipynb](https://github.com/Nicolas-Malgat/mlflow/blob/main/entrainement.ipynb "entrainement.ipynb")
**or**
-  Run [entrainement.py](https://github.com/Nicolas-Malgat/mlflow/blob/main/entrainement.py "entrainement.py")
```	python entrainement.py```

#### Run with .git path
It should be possible to run [entrainement.py](https://github.com/Nicolas-Malgat/mlflow/blob/main/entrainement.py "entrainement.py") project using the [github project link](https://github.com/Nicolas-Malgat/mlflow.git). But I didn't achieved to make it:
```
mlflow run https://github.com/Nicolas-Malgat/mlflow.git -P nb_epochs=15 -P nb_patience=4
```

#### MLflow UI
You can see registered runs using
```
mlflow ui
```
