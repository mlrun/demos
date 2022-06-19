import mlrun
from mlrun.datastore.sources import CSVSource
import mlrun.feature_store as fstore
from mlrun import code_to_function


project = mlrun.get_or_create_project(name='stocks', user_project=True, context="./")

news_set = fstore.FeatureSet("stocks_news",
                                 entities=[fstore.Entity("ticker")],
                                 timestamp_key='Datetime',
                                 description="stocks news feature set",engine="spark")


source = CSVSource("mycsv", path="news_input.csv")


news_set.graph \
    .to(name="print_dataframe", handler="print_dataframe")\
    .to(name="get_sentiment", handler="get_sentiment")



news_set.set_targets(with_defaults=True)
news_set.plot(rankdir="LR", with_targets=True)


my_func = code_to_function("func", filename="ingest_news_code.py", kind="spark")

my_func.with_driver_requests(cpu=1, mem="1G")
my_func.with_executor_requests(cpu=1, mem="1G")
my_func.with_igz_spark()
my_func.spec.use_default_image = True
my_func.spec.replicas = 2


config = fstore.RunConfig(local=True, function=my_func, handler="ingest_handler")
quotes_df = fstore.ingest(news_set, source, run_config=config)

