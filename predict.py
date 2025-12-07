import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import os

try:
    spark = SparkSession.builder.appName("Wine Quality Prediction").getOrCreate()
except Exception as e:
    print(f"Error initializing Spark session: {e}")
    sys.exit(1)

try:
    if len(sys.argv) != 2:
        print("Usage: spark-submit predict.py <test_csv_path> or python3 predict.py <test_csv_path>")
        sys.exit(1)
    test_path = sys.argv[1]
    MODEL_PATH = "model/wine_model"  # Local path for standalone/single EC2

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path '{MODEL_PATH}' not found. Run training first.")

    # Load model
    model = PipelineModel.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    spark.stop()
    sys.exit(1)

try:
    # Load and preprocess test data (pipeline handles features)
    test_data = spark.read.option("sep", ";").option("quote", "\"").csv(test_path, header=True, inferSchema=True)

    # Clean column names (strip extra quotes/spaces from header parsing)
    for col_name in test_data.columns:
        clean_name = col_name.strip('"').strip("'").strip()
        if clean_name != col_name:
            test_data = test_data.withColumnRenamed(col_name, clean_name)

    # If labels present, cast to double (for eval); assume 'quality' column exists for F1
    if "quality" in test_data.columns:
        test_data = test_data.withColumn("quality", col("quality").cast("double"))

    print(f"Test data loaded: {test_data.count()} rows")
    test_data.printSchema()
except Exception as e:
    print(f"Error loading test data from {test_path}: {e}. Ensure CSV format (semicolon-separated, quoted fields).")
    spark.stop()
    sys.exit(1)

try:
    # Predict
    predictions = model.transform(test_data)
    print("Predictions completed.")

    # Evaluate F1 if labels present
    if "quality" in predictions.columns:
        evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
        f1 = evaluator.evaluate(predictions)
        print(f"Test F1 Score: {f1:.4f}")
    else:
        print("No 'quality' column in test data; skipping F1 evaluation.")

    # Show predictions
    predictions.select("quality", "prediction", "probability").coalesce(1).write.mode("overwrite").option("header", "true").csv("predictions_output")
except Exception as e:
    print(f"Error during prediction or evaluation: {e}")
    spark.stop()
    sys.exit(1)

try:
    # Optional: Save predictions locally (for inspection)
    predictions.coalesce(1).write.mode("overwrite").option("header", "true").csv("predictions_output")
    print("Predictions saved to predictions_output/")
except Exception as e:
    print(f"Warning: Could not save predictions: {e}")

spark.stop()
