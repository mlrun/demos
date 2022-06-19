from mlrun.feature_store.api import ingest


def ingest_handler(context):
    ingest(mlrun_context=context)  # The handler function must call ingest with the mlrun_context


def print_dataframe(df):
    print("type of data frame {}".format(type(df)))
    print(df.show())
    return df


def print_row(row):
    print(row)
    return row


def get_sentiment(df):
    rdd = df.rdd
    rdd2 = rdd.flatMap(lambda x: print_row(x))
    for element in rdd2.collect():
        print(element)
    return df
