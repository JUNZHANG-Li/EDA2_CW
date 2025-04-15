import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (SparkSession.builder
         .appName("WeightedMeanPLDDTComputation")
         .getOrCreate())

# 1) Read all lines from every *.parsed file under /analysis_output/<subfolder>/
lines_df = spark.read.text("/analysis_output/*/*.parsed")

# ------------------------------------------------------------------
# EXTRACT MEAN PLDDT PER FILE
# ------------------------------------------------------------------
# Filter lines that start with '#' to capture "mean plddt: <value>"
plddt_lines_df = lines_df.filter(lines_df.value.startswith("#"))

# Extract the numeric mean plddt from the line text
# e.g. "#AF-P30855-F1-model_v4_search.tsv Results. mean plddt: 82.92179999999999"
plddt_lines_df = plddt_lines_df.withColumn(
    "plddt_value",
    F.regexp_extract(F.col("value"), r"mean plddt:\s*([0-9\.]+)", 1).cast("double")
)

# Keep track of the source file path
plddt_lines_df = plddt_lines_df.withColumn("path", F.input_file_name())

# Extract the immediate subfolder name (e.g. "ecoli_analysis_output")
plddt_lines_df = plddt_lines_df.withColumn(
    "subfolder",
    F.regexp_extract(F.col("path"), r"/analysis_output/([^/]+)/", 1)
)

# Convert subfolder name to a simpler organism label (e.g. "ecoli")
plddt_lines_df = plddt_lines_df.withColumn(
    "organism",
    F.regexp_replace(F.col("subfolder"), "_analysis_output", "")
)

# We only need (path, plddt_value, organism) for each file
plddt_per_file_df = plddt_lines_df.select("path", "plddt_value", "organism")

# ------------------------------------------------------------------
# EXTRACT TOTAL CATH COUNTS PER FILE
# ------------------------------------------------------------------
# Filter out header lines ('cath_id,count') and lines starting with '#'
count_lines_df = (lines_df
    .filter(~F.col("value").startswith("#"))
    .filter(~F.col("value").startswith("cath_id,count"))
    .filter(F.length("value") > 0))

# Split each data line by comma -> (cath_id, count)
split_col = F.split(count_lines_df["value"], ",")
count_df = count_lines_df.select(
    F.input_file_name().alias("path"),
    split_col.getItem(1).cast("int").alias("count")
)

# Sum the counts per file to get n_i
count_sums_df = count_df.groupBy("path").agg(F.sum("count").alias("count_sum"))

# ------------------------------------------------------------------
# COMBINE MEAN PLDDT WITH TOTAL COUNTS
# ------------------------------------------------------------------
# Join the two DataFrames on 'path'
combined_df = (count_sums_df
               .join(plddt_per_file_df, on="path", how="inner"))

# Now each row has:
#   path         <string>
#   count_sum    <int>    = n_i
#   plddt_value  <double> = p_i
#   organism     <string>

# ------------------------------------------------------------------
# AGGREGATE BY ORGANISM
# ------------------------------------------------------------------
# Compute sums needed for weighted average and weighted variance
agg_df = combined_df.groupBy("organism").agg(
    F.sum("count_sum").alias("sum_n_i"),  # Sum of n_i
    F.sum(F.col("count_sum") * F.col("plddt_value")).alias("sum_n_p_i"),
    F.sum(F.col("count_sum") * F.col("plddt_value") * F.col("plddt_value")).alias("sum_n_p_i2")
)

# ------------------------------------------------------------------
# CALCULATE WEIGHTED MEAN AND STD
# ------------------------------------------------------------------
# Weighted mean plddt = sum(n_i * p_i) / sum(n_i)
# Weighted variance   = [sum(n_i * p_i^2)/sum(n_i)] - (weighted_mean^2)
# Weighted std        = sqrt(weighted_variance)
summary_df = agg_df.select(
    F.col("organism"),
    F.format_number(
        (F.col("sum_n_p_i") / F.col("sum_n_i")), 
        2
    ).alias("mean plddt"),
    F.format_number(
        F.sqrt(
            (F.col("sum_n_p_i2") / F.col("sum_n_i")) -
            (F.col("sum_n_p_i") / F.col("sum_n_i")) ** 2
            ), 
        2
    ).alias("plddt std")
)

# ------------------------------------------------------------------
# WRITE RESULTS
# ------------------------------------------------------------------
# Output to a CSV directory named 'plDDT_means' in HDFS
summary_df.write.csv("plDDT_means", header=True)

spark.stop()





