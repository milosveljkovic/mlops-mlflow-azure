curl http://127.0.0.1:1234/invocations -H 'Content-Type: application/json; format=pandas-split' -d '{
    "columns": ["attitude.roll","attitude.pitch","attitude.yaw","userAcceleration.x","userAcceleration.y","userAcceleration.z","weight","height","age","gender","trial"],
    "data": [[0.476430,0.075142,0.465127, 0.442481,0.424474,0.507579,0.518519,0.655172,0.357143,1.0,0.400000]]
}'

curl "http://localhost:5000/api/2.0/preview/mlflow/model-versions/get?name=sk-learn-random-forest-reg-model&version=5"

curl http://3fd3f2a3-65e4-4372-828a-08f85bcc25f3.eastus.azurecontainer.io/score -H 'Content-Type: application/json; format=pandas-split' -d '{
    "columns": ["attitude.roll","attitude.pitch","attitude.yaw","userAcceleration.x","userAcceleration.y","userAcceleration.z","weight","height","age","gender","trial"],
    "data": [[0.476430,0.075142,0.465127, 0.442481,0.424474,0.507579,0.518519,0.655172,0.357143,1.0,0.400000]]
}'


cur_dir=$(pwd) && mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root $cur_dir/artifacts --host 0.0.0.0