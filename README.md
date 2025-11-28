# Project2_wine_quality
a wine quality prediction ML model in Spark over AWS
wine-quality-spark-ml/

├── README.md                    # Main documentation

├── QUICKSTART.md               # Quick reference

├── requirements.txt            # Dependencies

├── train_model.py              # Training script

├── predict_wine_quality.py     # Prediction script

├── Dockerfile                  # Container config

├── data/

│   ├── TrainingDataset.csv

│   └── ValidationDataset.csv

└── docs/

    └── setup_guide.pdf         # Comprehensive guide

Architecture Diagram.

                ┌─────────────────────────────────────────┐
                │                 Developer               │
                │   (Local PC / Lab Environment)          │
                └─────────────────────────────────────────┘
                                  │
                                  │ Flintrock CLI
                                  ▼
                ┌─────────────────────────────────────────┐
                │           Flintrock Controller           │
                │      (Local Machine running commands)    │
                └─────────────────────────────────────────┘
                                  │
                                  │ SSH / EC2 Launch
                                  ▼
       ┌────────────────────────────────────────────────────────────────┐
       │                         AWS EC2 Cluster                        │
       │                      (Flintrock Spark Cluster)                 │
       │                                                                │                                                                                                  │     ┌─────────────────────────────┐        ┌────────────────────┐
       │     │        Master Node          │        │     Worker Nodes   │
       │     │  (Spark Master + Driver)    │◄──────►│   (Executors x4)   │
       │     │                             │        │                    │
       │     │ - Receives spark-submit     │        │ - Run tasks        │
       │     │ - Schedules jobs            │        │ - Store shuffle    │
       │     │ - Web UI :8080              │        │ - Partitions data  │
       │     └─────────────────────────────┘        └────────────────────┘
       │                      │
       │                      │ S3A I/O
       │                      ▼
       └────────────────────────────────────────────────────────────────┘
                                  │
                                  │ Read/Write via Hadoop S3A
                                  ▼
         ┌────────────────────────────────────────────────────────────┐
         │                            Amazon S3                       │
         │                                                            │
         │   ┌────────────────────────┐     ┌────────────────────────┐│
         │   │   Input Data Bucket    │     │     Model Storage       ││
         │   │ (ValidationDataset.csv)│     │s3://ml-spark-project2/..││
         │   └────────────────────────┘     └────────────────────────┘│
         │                                                            │
         └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
           ┌───────────────────────────────────────────────────────────┐
           │                  Output & Evaluation                      │
           │  - Training metrics (F1 score)                            │
           │  - Predictions on new CSV                                 │
           │  - Model saved back to S3                                 │
           └───────────────────────────────────────────────────────────┘
