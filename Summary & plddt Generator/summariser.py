from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import input_file_name, regexp_extract

spark = (SparkSession.builder
         .appName("CATHAnalysis")
         .config("spark.ui.showConsoleProgress", "true")
         .getOrCreate())

# Read in all .parsed files from all subdirectories under /analysis_output
lines_df = (
    spark.read.text("/output/*/*.parsed")
    .withColumn("subdir", regexp_extract(input_file_name(), r"/analysis_output/([^/]+)/", 1))
)

# Filter out unwanted lines
filtered_df = (
    lines_df
    .filter(~F.col("value").startswith("#"))
    .filter(~F.col("value").startswith("cath_id,count"))
    .filter(F.length("value") > 0)
)

# Split lines and parse columns
split_col = F.split(filtered_df["value"], ",")
parsed_df = filtered_df.select(
    "subdir",
    split_col.getItem(0).alias("cath_id"),
    split_col.getItem(1).cast("int").alias("count")
)

# Group by subdir and cath_id, summing counts
summary_df = parsed_df.groupBy("subdir", "cath_id").agg(F.sum("count").alias("count"))

# Determine each distinct subdirectory
distinct_subdirs = [row[0] for row in summary_df.select("subdir").distinct().collect()]

# Write out a separate CSV for each subdirectory
for subdir in distinct_subdirs:
    subdir_df = summary_df.filter(F.col("subdir") == subdir).select("cath_id", "count")
    output_dir = subdir.replace("_analysis_output", "_cath_summary")
    subdir_df.write.csv(output_dir, header=True)

spark.stop()

