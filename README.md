"# mlflow" 

https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html

conda env create

git clone https://github.com/mlflow/mlflow

python examples\sklearn_elasticnet_wine\train.py

mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0

mlruns\0\44e40ab200fc4e58af1f131f419aacb5\artifacts\model

mlflow models serve -m mlruns\0\44e40ab200fc4e58af1f131f419aacb5\artifacts\model -p 1234
    curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{\"columns\":[\"alcohol\", \"chlorides\", \"citric acid\", \"density\", \"fixed acidity\", \"free sulfur dioxide\", \"pH\", \"residual sugar\", \"sulphates\", \"total sulfur dioxide\", \"volatile acidity\"],\"data\":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations
