# ⚡ Wine Quality Predictor (Cloud Programming Assignment 2)

This project builds and deploys a Spark ML regression model that predicts wine quality based on a combination of numerical and categorical features.
Model training is performed in a distributed Spark standalone cluster (Master + Worker nodes) running on Amazon EC2.
After training, predictions can be executed either:

Directly on a Spark cluster, or

Inside a lightweight Docker container that packages Spark, Hadoop (for S3A support), and the prediction script.

This setup demonstrates distributed data processing, containerized inference, and cloud-based execution using Apache Spark, Hadoop, and Docker.


---
All the commands and code for training and prediction require following the Project/Directory structure.

## 📁 Project Structure

```
wine-quality/
├── data/
│       |
│       ├── TrainingDataset.csv
│       └── ValidationDataset.csv
├── model/
│   └── wine_model/         # Final trained model
├── spark-docker/
│   ├── Dockerfile
│   └── ...
├── training/
│   ├── train.py              # Trains and saves best model
│   └── predict.py            # Predicts on unseen data
└── README.md                 # Project instructions (This file)
```

---


## 🌐 Cloud Setup Instructions (AWS EC2)

 1. Launch 4 EC2 Instances (Ubuntu)

   * 1 master + 3 workers (recommended 't2.medium' or larger like m5.large) with Ubuntu Server 24.04 LTS (HVM) AMI
      (Note: Bigger and better machines reduce the training and execution time but have no effect of RMSE)
   * Choose vockey Key pair
   * Ensure all are on the same VPC for private IP communication
   * A storage of 16GB is recommended for Master and 8GB for all the workers
   * Inbound Rules:
      Allow all traffic between the EC2s VPC (Subnet of the instances x.x.0.0/16).
      Allow SSH traffic from your device's Public IP (e.g. My Laptop's Public IP) on all instances.
      Allow SSH traffic from Master's Private IP to all Workers.
      Allow Custom TCP traffic on port 8080 for Master EC2 (Spark UI) or include in the security group common to all instances.

 2. Install Prerequisites on All Instances

```bash
sudo apt update
sudo apt install -y openjdk-17-jdk python3 python3-pip git
pip3 install pandas
```

 3. Install Apache Spark (on all instances) (PWD: home/ubuntu/)

```bash
wget https://downloads.apache.org/spark/spark-4.0.1/spark-4.0.1-bin-hadoop3.tgz
tar -xvzf spark-4.0.1-bin-hadoop3.tgz
sudo mv spark-4.0.1-bin-hadoop3 wine-quality/spark
```

 4. Configure Environment Variables

Append to `~/.bashrc`:

# All nodes (Master + Workers):

   ```bash
   export SPARK_HOME=~/wine-quality/spark
   export PATH=$SPARK_HOME/bin:$PATH
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
   ```

Then run `source ~/.bashrc`

 5. Set Up SSH Access

* Create SSH key on master: `ssh-keygen`
   Command: `ssh-keygen -t rsa`
* Copy pub key to all workers: `ssh-copy-id <worker_ip>` (Requires SSHPASS)
   Command: `ssh-copy-id ubuntu@<worker-private-ip>`
   Alternative: Manually copy and paste the pub key in .ssh/authorized_keys of each Worker
                (Note: Add this new pub key in a new line and do not erase any existing keys like vockey which is required to SSH into the machine from your device)
* Add workers' Private IPs to `$SPARK_HOME/conf/workers` one per line


 6. Start the Cluster from Master

```bash
$SPARK_HOME/sbin/start-all.sh
```

Spark UI is available at:
[`http://<master_public_ip>:8080`](http://ip-172-31-36-193.ec2.internal:4040)

Can confirm using the Web UI or logging into each worker and typing the command `jps`


---


## 🔨 Model Training

 🔹 Format (PWD: wine-quality/)

```bash
spark-submit training/train.py <path_to_training_csv> <path_to_validation_csv>
```

 🔹 In my case (PWD: wine-quality/)

```bash
spark-submit training/train.py data/TrainingDataset.csv data/ValidationDataset.csv
```

* Can change the inputs but relative path to the file (train.py, Validation dataset) needs to be input
* Saves best model to `model/wine_model/`
* Automatically selects the regressor with the lowest RMSE (e.g., Linear Regression, GLR, RF, etc.)


---


## 📈 Model Prediction to confirm the training of the model

 🔹 Spark-based Prediction (Cluster or Docker) (PWD: wine-quality/)

```bash
spark-submit training/predict.py <path_to_validation_csv>
```
    🔹 In my case (PWD: wine-quality/)

```bash
spark-submit training/predict.py data/ValidationDataset.csv
```

 🔹 Python-only Prediction (Single EC2 instance) (PWD: wine-quality/)

```bash
python3 training/predict.py <path_to_test_csv>
```
Install PySpark manually if not installed: pip3 install pyspark
If in case of environment restriction:     pip3 install --break-system-packages pyspark

    🔹 In my case (PWD: wine-quality/)

```bash
python3 training/predict.py data//ValidationDataset.csv 
```


* Can change the inputs but relative path to the file (predict.py, Validation dataset) needs to be input
* Outputs the RMSE and preview of predictions.
* Uses the saved model from `model/wine_model/`.


---


## 🐳 Docker-based Prediction

 1. Move into the docker context

```bash
cd wine-quality/
```

 2. Build the image (PWD: wine-quality/)

```bash
docker build -t jchoi1986/project2:latest -f spark-docker/Dockerfile .
 

To build a new image and remove the old image, run the following commands:
   `sudo docker build --no-cache -t wine-predictor -f spark-docker/Dockerfile .`
   `docker image prune` <---- This command removes untagged images (<none> tag)

```

 3. Run the container (PWD: wine-quality/)

🔹 Format

```bash
docker run --rm -v "$(pwd)/data:/app/data" jchoi1986/project2:latest data/ValidationDataset.csv

```

```bash
sudo docker run --rm -v "$(pwd)/data:/app/data" wine-predictor data/ValidationDataset.csv
```
In my case, inside the container, the path data/ValidationDataset.csv on my host becomes data/ValidationDataset.csv because of the volume mount. 
This is why the prediction script expects just data/ValidationDataset.csv.

 4. Push to Docker Hub

```bash
docker login
docker tag jchoi1986/project2:latest jchoi1986/project2:latest
docker push jchoi1986/project2:latest

docker image ls
docker tag <latest> jchoi1986/project2:latest
docker push jchoi1986/project2:

example output:

The push refers to repository [docker.io/jchoi1986/project2]
f6d111e1ae1b: Pushed
51c251f705a4: Pushed
20c3873351c0: Pushed
73974f74b436: Pushed
latest: digest: sha256:0ab0a330227c9b0432f4c739de5caa48957109e17ea948001c930392ef1631e9 size: 1168

```

---


## 🔍 How to Replicate

 🔹 Format

```bash
sudo docker run -v "$(pwd)/<test_folder>:/app/data" jchoi1986/wine-predictor data/<test_file>.csv
```
An optional `--rm` tag can be used along to automaticall remove the container after it exits (Cleaner testing during development)

---


## 🧠 Code Attribution  

 Code Written from Scratch

* Project structure
* Data wrangling logic
* Core working logic of train.py and predict.py
* All bash and setup scripting
* Dockerfile organization and testing logic

 Code Generated with ChatGPT

* PySpark pipeline scaffolding for `train.py` and `predict.py`
* Helped identify proper transformations (Indexing, Scaling)
* Styling and formatting the README file

 Code Adapted from ChatGPT Suggestions

* Modified evaluation logic and fixed model saving issues
* Refined cross-validation selection
* Aligned Spark MLlib flow with cluster configurations


---


## 💬 Experience with ChatGPT

ChatGPT was highly useful throughout the project, especially for brainstorming approaches, validating my reasoning, and quickly generating scaffolding code for PySpark and Docker. It accelerated development by reducing boilerplate work and helping structure the PySpark pipeline, model logic, and Dockerfile layout.

However, the real challenges—environment setup, distributed Spark debugging, EC2 and SSH configuration, handling version mismatches, PySpark serialization issues, and resolving Docker volume and path problems—still required extensive manual troubleshooting and deeper technical understanding. ChatGPT could guide the direction, but it couldn’t replace hands-on debugging or contextual decision-making.

Overall, ChatGPT served as a fast ideation and debugging companion rather than an end-to-end solution. It improved productivity and clarified concepts, but successfully productionizing the full prediction pipeline ultimately depended on careful manual iteration, testing, and configuration tuning.

---


## ✅ Final Deliverables

GitHub Code: [https://github.com/JC1686/Project2_wine_quality/tree/main]
Docker Image: [https://hub.docker.com/repository/docker/jchoi1986/project2/general]
