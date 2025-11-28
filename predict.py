import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

if len(sys.argv) != 2:
    print("Usage: spark-submit predict.py <input_csv_path>")
    sys.exit(1)

spark = SparkSession.builder.appName("WinePredict").getOrCreate()
input_path = sys.argv[1]

# Load full pipeline model (includes assembler, optional scaler, and LR; from S3)
model_path = "s3a://ml-spark-project2/models/wine-model"
model = PipelineModel.load(model_path)

# Load test data (assumes same schema as training: features + 'quality' column)
test = spark.read.option("header", "true").csv(input_path)
# Cast quality to int (for multiclass labels 0-9)
test = test.withColumn("quality", col("quality").cast(IntegerType()))

# Predict (pipeline handles preprocessing)
preds = model.transform(test)

# Evaluate F1 (multiclass weighted average)
evaluator = MulticlassClassificationEvaluator(
    labelCol="quality", 
    predictionCol="prediction", 
    metricName="f1"
)
f1 = evaluator.evaluate(preds)
print(f"Prediction F1 Score: {f1}")

spark.stop()