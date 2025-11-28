from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("WineQualityModelTraining") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

S3_BUCKET = "ml-spark-project2"
TRAIN_PATH = f"s3://{S3_BUCKET}/TrainingDataset.csv"
VAL_PATH = f"s3://{S3_BUCKET}/ValidationDataset.csv"
MODEL_SAVE_PATH = f"s3://{S3_BUCKET}/models/wine-model"   # <<< match predict.py

train = spark.read.csv(TRAIN_PATH, header=True, inferSchema=True)
val = spark.read.csv(VAL_PATH, header=True, inferSchema=True)

features = [c for c in train.columns if c != 'quality']

assembler = VectorAssembler(inputCols=features, outputCol="rawFeatures")
scaler = StandardScaler(inputCol="rawFeatures", outputCol="features", withStd=True, withMean=True)
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)

# Build FULL pipeline
pipeline = Pipeline(stages=[
    assembler,
    scaler,
    lr
])

# Prepare training and validation sets
train = train.select(col("quality").cast("double").alias("label"), *features)
val = val.select(col("quality").cast("double").alias("label"), *features)

# Hyperparameter tuning
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 0.5]).build()
evaluator = MulticlassClassificationEvaluator(metricName="f1")

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=4
)

# Train
cvModel = cv.fit(train)
bestModel = cvModel.bestModel

# Validate
preds = bestModel.transform(val)
f1 = evaluator.evaluate(preds)
print(f"Best Validation F1: {f1:.4f}")

# Save *full pipeline*
bestModel.save(MODEL_SAVE_PATH)
print(f"Saved full pipeline model to {MODEL_SAVE_PATH}")

spark.stop()
