import os
import mlrun.feature_store as fstore
from mlrun.datastore.targets import (
    ParquetTarget,
    RedisNoSqlTarget,
)

from mlrun.datastore.sources import (
    StreamSource,
    KafkaSource
)


def get_redis_uri():
    if os.environ.get('REDIS_IP'):
        redis_ip = os.environ.get('REDIS_IP')
    else:
        raise Exception("empty environment variable REDIS_IP ,cannot create RedisNoSQLTarget")
    if os.environ.get('REDIS_PORT'):
        redis_port = os.environ.get('REDIS_PORT')
    redis_uri = 'redis://{}:{}'.format(redis_ip, redis_port)
    return redis_uri

def get_kafka_stream(topic):
    if os.environ.get('KAFKA_IP'):
        kafka_ip = os.environ.get('KAFKA_IP')
    else:
        raise Exception("empty environment variable KAFKA_IP ,cannot create Kafka Stream")
    if os.environ.get('KAFKA_PORT'):
        kafka_port = os.environ.get('KAFKA_PORT')
    kafka_uri = 'kafka://{}:{}/{}'.format(kafka_ip, kafka_port,topic)
    return kafka_uri



def set_feature_set_targets_by_environment(feature_set: fstore.FeatureSet):
    # if running on iguazio platform
    if os.environ.get('V3IO_ACCESS_KEY'):
        feature_set.set_targets()
    # running on mlrun community edition
    else:
        redis_uri = get_redis_uri()
        targets = [ParquetTarget(name='transactions', partitioned=True, partition_cols=["timestamp"]),
                   RedisNoSqlTarget(name="write", path=redis_uri)]
        feature_set.set_targets(targets=targets, with_defaults=False)

        
def get_stream_uri(project_name, stream_suffix_path):
    # if running on iguazio platform
    if os.environ.get('V3IO_ACCESS_KEY'):
        transaction_stream = 'v3io:///projects/{}/streams/{}'.format(project_name, stream_suffix_path)
    else:
        transaction_stream = get_kafka_stream("transaction")
    return transaction_stream     

def get_source_by_environment(stream_path):
    if os.environ.get('V3IO_ACCESS_KEY'):
        source = StreamSource(path=stream_path, key_field='source',
                                                      time_field='timestamp')
    else:
        if os.environ.get('KAFKA_IP'):
            kafka_ip = os.environ.get('KAFKA_IP')
        else:
            raise Exception("empty environment variable KAFKA_IP ,cannot create Kafka Stream")
        if os.environ.get('KAFKA_PORT'):
            kafka_port = os.environ.get('KAFKA_PORT')
        source = KafkaSource(brokers="{}:{}".format(kafka_ip, kafka_port), topics="transactions",
                                                     key_field='source', time_field='timestamp')
    return source
