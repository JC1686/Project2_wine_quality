from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean as spark_mean
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import sys

try:
    # Step 1: Initialize Spark Session (with MLlib tuning from conf)
    spark = SparkSession.builder.appName("Wine Quality Prediction") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
except Exception as e:
    print(f"Error initializing Spark session: {e}")
    sys.exit(1)

try:
    # Get paths from command-line args (local or S3)
    if len(sys.argv) < 3:
        print("Usage: spark-submit train.py <train_csv_path> <val_csv_path>")
        sys.exit(1)
    TRAIN_PATH = sys.argv[1]
    VAL_PATH = sys.argv[2]
    MODEL_SAVE_PATH = "model/wine_model"  # Local for standalone; override with arg if needed

    # Load Data (Wine dataset is semicolon-separated with quoted fields)
    train_data = spark.read.option("sep", ";").option("quote", "\"").csv(TRAIN_PATH, header=True, inferSchema=True)
    validation_data = spark.read.option("sep", ";").option("quote", "\"").csv(VAL_PATH, header=True, inferSchema=True)

    # Clean column names (strip extra quotes/spaces from header parsing)
    for col_name in train_data.columns:
        clean_name = col_name.strip('"').strip("'").strip()
        if clean_name != col_name:
            train_data = train_data.withColumnRenamed(col_name, clean_name)
    for col_name in validation_data.columns:
        clean_name = col_name.strip('"').strip("'").strip()
        if clean_name != col_name:
            validation_data = validation_data.withColumnRenamed(col_name, clean_name)

    # Print schema for verification
    print("Train Schema:")
    train_data.printSchema()
    print(f"Train rows: {train_data.count()}")
    print(f"Validation rows: {validation_data.count()}")
except Exception as e:
    print(f"Error loading data from {TRAIN_PATH}/{VAL_PATH}: {e}. Ensure paths exist and format is CSV.")
    spark.stop()
    sys.exit(1)

try:
    # Step 1.5: Data Preprocessing (Handle missing values; keep quality as int)
    # Assume columns: fixed_acidity, volatile_acidity, ..., alcohol, quality (target)
    feature_columns = [x for x in train_data.columns if x != "quality"]
    target_col = "quality"

    if target_col not in train_data.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Handle missing values: Impute numerical features with mean
    for col_name in feature_columns:
        if col_name not in train_data.columns:
            raise ValueError(f"Feature column '{col_name}' not found in dataset.")
        mean_val = train_data.select(spark_mean(col(col_name)).alias("mean")).collect()[0]["mean"]
        train_data = train_data.fillna({col_name: mean_val})
        validation_data = validation_data.fillna({col_name: mean_val})
    print(f"Preprocessed: Missing values imputed with column means.")
except Exception as e:
    print(f"Error during data preprocessing: {e}")
    spark.stop()
    sys.exit(1)

try:
    # Step 2: Build Feature Vector (Assembler + Scaler for normalization)
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="rawFeatures")
    scaler = StandardScaler(inputCol="rawFeatures", outputCol="features", withStd=True, withMean=True)

    # Step 3: Train Model (Logistic Regression with pipeline and CV tuning)
    lr = LogisticRegression(featuresCol="features", labelCol=target_col, maxIter=50, family="multinomial")  # Multiclass

    # Pipeline: Assembler -> Scaler -> LR
    pipeline = Pipeline(stages=[assembler, scaler, lr])

    # Hyperparameter tuning (regParam for regularization)
    paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 0.5]).build()
    evaluator = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="f1")
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4  # Tune based on cores
    )

    # Fit on train data
    print("Training model with Cross-Validation...")
    cvModel = cv.fit(train_data)
    bestModel = cvModel.bestModel
except Exception as e:
    print(f"Error during model training or tuning: {e}. Check data quality or hyperparameters.")
    spark.stop()
    sys.exit(1)

try:
    # Step 4: Validate Model
    predictions = bestModel.transform(validation_data)  # Pipeline handles assembly/scaling
    f1_score = evaluator.evaluate(predictions)
    print(f"Best Validation F1 Score: {f1_score:.4f}")

    # Show sample predictions
    predictions.select(target_col, "prediction", "probability").show(10)
except Exception as e:
    print(f"Error during model validation: {e}")
    spark.stop()
    sys.exit(1)

try:
    # Save full pipeline model
    bestModel.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Error saving model to {MODEL_SAVE_PATH}: {e}. Ensure write permissions.")
    spark.stop()
    sys.exit(1)

spark.stop()
