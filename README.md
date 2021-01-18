https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
git clone https://github.com/mlflow/mlflow
python examples\sklearn_elasticnet_wine\train.py

mlflow run https://github.com/Nicolas-Malgat/mlflow.git
mlflow run https://github.com/Nicolas-Malgat/mlflow.git -P nb_epochs=15 -P nb_patience=4  

jupyter nbconvert 1_entrainement.ipynb --to python

conda env create
