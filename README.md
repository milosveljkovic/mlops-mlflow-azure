# mlops-mlflow-azure (Activity Tracking)
Machine Learning Operation platform which retrain and redeploy ML model automaticly on azure using CRI.

## Description of MLOps system

Activity Tracking is a system based on fastAPI microservices that, together with the corresponding Azure services, enables continuous monitoring, deployment and automatic retraining of ML models. The entire platform is built around the MLFlow ecosystem.

The Activity Tracking system can be broken down into several smaller components: Data, ML and Deploy subsystems.
Dataset can be found here: [dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

The `Data subsystem` focuses on the data itself and its nature. With the arrival of each new data, the `Data subsystem` performs an appropriate analysis with the intention of detecting changes `(drift)` in the data. Any change in any feature of the data drastically affects the precision of the ML model, so at that moment it is necessary to train the model again but with new drift data. Drift detection is based on the `ADWIN` method, and the implementation of the `Drift Detector` microservice itself was done using the open source tool river.

In the case of detecting a drift in the data, the corresponding action (HTTP request) is triggered, which activates the `ML subsystem`, which consists of two components: `Model Retraining` and `MLFlow`. Each detection of drift in the data activates the `Model Retraining` microservice that trains multiple models. For each trained model, parameters are logged in the `MLFlow Tracking` component so that the system user can monitor the evolution of the ML model as a consequence of the evolution of the data.

Mlfow Tracking:

![image12](https://user-images.githubusercontent.com/47954575/189479545-3aa82e21-626d-4b7d-9310-c0cbdf2ddb9b.png)

The model that shows the best performance with new data is packed into a python pickle file after retraining and registered in the `MLFlow Registry`.

Mlflow Registry:

![image13](https://user-images.githubusercontent.com/47954575/189479569-31d7cabf-4b43-49b3-ab5b-ce84042d5ec5.png)

After the model is registered in the registry, it needs to be deployed. The last component of the system is the `Deploy subsystem`, which is in charge of interacting with `Azure` services as well as deploying models. In order for the entire `Deploy system` to be possible, it is necessary to first create a suitable `ML workspace` on Azure, which meant creating the following resources: `Container Registry`, `Azure Blob`, `Azure ML Workspace` as well as `Container Instances`. As mentioned, after registering the model, the deployment of the model follows, which implies the interaction of the previously mentioned Azure resources with the Model Deploy microservice. The first step involves generating the appropriate Docker Flask image to be registered with the `Azure Container Registry`.

Azure Container Registry:

![image8](https://user-images.githubusercontent.com/47954575/189479612-64fbd584-dd8c-4a18-9ae6-1149d11e4a2a.png)

After a successfully registered Docker image, it is necessary to create an Azure Container Instance (ACI) that will automatically create a suitable endpoint that can be used immediately.

ACI:

![image7](https://user-images.githubusercontent.com/47954575/189479648-3a07208e-01e9-4edc-9e1d-8443a97575ca.png)

Architectural diagram:

![image4](https://user-images.githubusercontent.com/47954575/189479369-b0a1cf20-a8d6-4c5a-ade4-e0cca31a2b94.png)
