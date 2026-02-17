from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, min, max

# -----------------------------
# Spark Session
# -----------------------------
spark = SparkSession.builder \
    .appName("Churn-Validation") \
    .getOrCreate()

# -----------------------------
# Load Silver Data
# -----------------------------
df = spark.read.parquet("data/silver/churn_clean.parquet")

print("\n--- SILVER SCHEMA ---")
df.printSchema()

# -----------------------------
# Basic Row Count
# -----------------------------
print("\n--- ROW COUNT ---")
print(df.count())

# -----------------------------
# Churn Distribution
# -----------------------------
print("\n--- CHURN DISTRIBUTION ---")
df.groupBy("churn").count().show()

# -----------------------------
# Numerical Summary
# -----------------------------
print("\n--- BILL AMOUNT STATS ---")
df.select(
    min("bill_amount").alias("min_bill"),
    avg("bill_amount").alias("avg_bill"),
    max("bill_amount").alias("max_bill")
).show()

print("\n--- CALL MINUTES STATS ---")
df.select(
    min("call_minutes").alias("min_minutes"),
    avg("call_minutes").alias("avg_minutes"),
    max("call_minutes").alias("max_minutes")
).show()

# -----------------------------
# Data Quality Checks
# -----------------------------
print("\n--- NEGATIVE VALUE CHECK ---")
df.filter(
    (col("bill_amount") < 0) |
    (col("call_minutes") < 0) |
    (col("data_usage_gb") < 0)
).show()

print("\n✅ Data validation completed successfully")

spark.stop()
