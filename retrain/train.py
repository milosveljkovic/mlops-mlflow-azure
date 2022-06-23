import os
import warnings
import sys
import azureml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import subprocess
import logging
from azureml.core import Workspace
import mlflow.azureml
from azureml.core.webservice import AciWebservice, Webservice
import uuid
from  mlflow.tracking import MlflowClient
client = MlflowClient()

workspace_name = "azure-mlops-workspace"
workspace_location="East US"
resource_group = "azure-mlops"
subscription_id = "subscription_id"
workspace = Workspace.create(name = workspace_name,
                            location = workspace_location,
                            resource_group = resource_group,
                            subscription_id = subscription_id,
                            exist_ok=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

run_id = ""
model_name = ""

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name not in ['act','id']:
          max_value = df[feature_name].max()
          min_value = df[feature_name].min()
          result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def build_image(run_id, model_name):
    model_uri = f"models:/{model_name}/latest"
    
    model_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri,
                                                        workspace=workspace,
                                                        model_name="motion_detection",
                                                        image_name="model",
                                                        description="Some description of model",
                                                        synchronous=False)
    print("HEllo")
    model_image.wait_for_creation(show_output=True)

    aci_service_name = "motion-detection-model-runid-"+ run_id #str(uuid.uuid4())[:4]
    aci_service_config = AciWebservice.deploy_configuration()
    aci_service = Webservice.deploy_from_image(name=aci_service_name,
                                            image=model_image,
                                            deployment_config=aci_service_config,
                                            workspace=workspace)

    aci_service.wait_for_deployment(show_output=True)
    print(aci_service.get_logs())
    print(aci_service.scoring_uri)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    warnings.filterwarnings("ignore")
    # experiments = client.list_experiments() # returns a list of mlflow.entities.Experiment
    # run = client.create_run(experiments[0].experiment_id) # returns mlflow.entities.Run
    '''
    Read data from specified source.
    Source get as a param!
    '''
    try:
        df = pd.read_csv("/home/milosv/Desktop/my-github/mlops/retrain/motion_final_part_0.csv")
        # df = pd.read_csv("/home/milosv/Desktop/mlflow-fresh/project/monorepo/motion_final_part_0.csv")
        # logger.info("Dataset {} has been sucesfully read!","/home/milosv/Desktop/mlflow-fresh/project/monorepo/motion_final_part_0.csv")
        logger.info("Dataset {} has been sucesfully read!","/home/milosv/Desktop/my-github/mlops/retrain/motion_final_part_0.csv")
        print(df.head()) ### DELETE THIS
    except Exception as e:
        logger.exception("Unable to download dataset for retraining, check your internet connection. Error: %s", e)

    df=normalize(df)
    print(df.head()) ### DELETE THIS

    x = df.drop(['act','id'], axis = 1)
    y = df['act']

    np.random.seed(42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    with mlflow.start_run():
        models = { 
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": GaussianNB()
            }

        # Kreiranje funkcije za treniranje i ocenjivanje modela
        def fit_and_score(models, x_train, x_test, y_train, y_test):
          np.random.seed(42)
          model_scores = {}
          max_model_score = -1
          for name, model in models.items():
              model.fit(x_train, y_train)
              model_scores[name] = model.score(x_test, y_test)
              predicted_qualities = model.predict(x_test)
              (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
              mlflow.log_metrics({name:model_scores[name],name+"-rmse":rmse,name+"-r2":r2,name+"-mae": mae})
              if max_model_score < model_scores[name]:
                max_model_score = model_scores[name]
                new_model = model
          return new_model

        my_new_model = fit_and_score(models = models,
                                    x_train = x_train,
                                    x_test = x_test,
                                    y_train = y_train,
                                    y_test = y_test)
        
        # mlflow.sklearn.log_model(my_new_model, "model")
        signature = infer_signature(x_train, my_new_model.predict(x_train))
        # mlflow.sklearn.save_model(my_new_model, "artifacts/0/"+mlflow.active_run().info.run_id,
        # serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,signature=signature)

        mlflow.sklearn.log_model(sk_model=my_new_model,
                                artifact_path="artifacts",
                                registered_model_name="motion_detection",
                                signature=signature, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE )

        ## sleep some time

        # build_image(mlflow.active_run().info.run_id, "motion_detection")