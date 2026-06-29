# Databricks notebook source
# MAGIC %md
# MAGIC # Reddit Topics Lakehouse — Data Profiling
# MAGIC
# MAGIC Exploratory profiling over the **silver `events`** table. This notebook is
# MAGIC read-only: it inspects volumes, distributions, sparsity and time coverage
# MAGIC to inform DQ thresholds and ML feature choices. It does **not** mutate any
# MAGIC table. Set the catalog widget to point at the desired environment.

# COMMAND ----------

dbutils.widgets.text("catalog", "reddit_dev", "Unity Catalog")
dbutils.widgets.text("silver_schema", "silver", "Silver schema")
catalog = dbutils.widgets.get("catalog")
silver_schema = dbutils.widgets.get("silver_schema")
events_table = f"{catalog}.{silver_schema}.events"
print("Profiling:", events_table)

# COMMAND ----------

from pyspark.sql import functions as F

events = spark.read.table(events_table)
events.cache()
total = events.count()
print(f"Total events: {total:,}")
events.printSchema()

# COMMAND ----------

# MAGIC %md ## Event type & topic mix

# COMMAND ----------

display(
    events.groupBy("topic", "event_type")
    .agg(F.count("*").alias("events"), F.avg("score").alias("avg_score"))
    .orderBy("topic", "event_type")
)

# COMMAND ----------

# MAGIC %md ## Null / sparsity profile across key columns

# COMMAND ----------

key_cols = ["event_id", "subreddit", "created_ts", "text", "score", "text_length"]
exprs = [
    F.round(F.sum(F.col(c).isNull().cast("int")) / F.lit(total), 4).alias(c)
    for c in key_cols
]
display(events.select(*exprs))

# COMMAND ----------

# MAGIC %md ## Score distribution (percentiles)

# COMMAND ----------

display(
    events.select(
        F.expr("percentile_approx(score, array(0.01,0.25,0.5,0.75,0.9,0.99))").alias("score_pcts"),
        F.min("score").alias("min_score"),
        F.max("score").alias("max_score"),
        F.avg("score").alias("mean_score"),
    )
)

# COMMAND ----------

# MAGIC %md ## Text length distribution (drives TF-IDF vocab sizing)

# COMMAND ----------

display(
    events.select(
        F.expr("percentile_approx(word_count, array(0.5,0.9,0.99))").alias("word_count_pcts"),
        F.avg("word_count").alias("mean_words"),
        F.max("word_count").alias("max_words"),
    )
)

# COMMAND ----------

# MAGIC %md ## Temporal coverage & daily volume

# COMMAND ----------

display(
    events.groupBy("created_date")
    .agg(F.count("*").alias("events"))
    .orderBy("created_date")
)

# COMMAND ----------

# MAGIC %md ## Top subreddits by volume

# COMMAND ----------

display(
    events.groupBy("topic", "subreddit")
    .agg(F.count("*").alias("events"), F.avg("score").alias("avg_score"))
    .orderBy(F.desc("events"))
    .limit(25)
)

# COMMAND ----------

events.unpersist()
print("Profiling complete.")
