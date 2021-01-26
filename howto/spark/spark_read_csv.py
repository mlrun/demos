from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from mlrun import get_or_create_ctx

context = get_or_create_ctx("spark-function")

# build spark session
spark = SparkSession.builder.appName("Spark job").getOrCreate()

# read csv
df = spark.read.load('howto/spark-operator/iris.csv', 
                     format="csv", 
                     sep=",", 
                     header="true")

# sample for logging
df_to_log = df.describe().toPandas()

# log final report
context.log_dataset("df_sample", 
                     df=df_to_log,
                     format="csv")
spark.stop()