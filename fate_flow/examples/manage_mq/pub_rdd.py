import pika, json
import pyspark

print("[x] send hello world!")

exchange_name = "testing-exchange"
queue_name = "testing-queue"

def HandlePartition(index, iterator):
    msg = ""
    elements = []
    for element in iterator:
        msg += ' ' + str(element) + ','
        elements.append(element)

    credentials = pika.PlainCredentials('federate-test', '123456')
    connection = pika.BlockingConnection(pika.ConnectionParameters("localhost", 5672, "federated", credentials))

    channel = connection.channel()

    channel.exchange_declare(exchange=exchange_name, durable=True)
    channel.queue_declare(queue=queue_name, durable=True)
    channel.queue_bind(exchange=exchange_name, queue=queue_name)
    
    channel.basic_publish(exchange=exchange_name, routing_key=queue_name, body=json.dumps(elements))

    return elements



num_slice = 4 

sc = pyspark.SparkContext(appName="spark_redis_producer")
# rdd_test = sc.parallelize([i for i in range(1000)], num_slice)
text_file = sc.textFile("/home/luke/Document/github/spark/README.md")

counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)

rdd_test = sc.parallelize(counts.collect(), num_slice)
new_rdd_test = rdd_test.mapPartitionsWithIndex(HandlePartition)
new_rdd_test.count()

