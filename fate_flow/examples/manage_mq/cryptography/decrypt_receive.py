import pyspark, pika
from src import decrypt_blob, encrypt_blob

def HandlePartition(messages, private_key):
    for encrypted_message in messages:
        if len(encrypted_message) != 0:
            print("Before decrypted: ", encrypted_message[:10])
            msg = decrypt_blob(encrypted_message, private_key)
            print("After decrypted: ", msg[:10])
            return [msg]
    return []

sc = pyspark.SparkContext(appName="spark_redis_producer")

fd = open("private_key.pem", "rb")
private_key = fd.read()
fd.close()

lines = []
with open("../tmp_share", "r") as f:
    lines = f.readlines()

job_id = lines[2].replace('\n', "")
user = lines[0].replace('\n', "")
password = lines[1].replace('\n', "")

credentials = pika.PlainCredentials(user, password)
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost", 5673, job_id, credentials))
channel = connection.channel()

union_rdd = sc.emptyRDD()

count = 0
for method, properties, body in channel.consume(queue="receive-{}".format(job_id)):

    rdd_test = sc.emptyRDD()
    if len(body) != 0:
        rdd_test = sc.parallelize([body])
        ## print("#########This is the parallelize: ", rdd_test.collect()[:10])
        rdd_test = rdd_test.mapPartitions(lambda x: HandlePartition(x, private_key))

    union_rdd = union_rdd.union(rdd_test)
    print("This is the collections result: ", len(union_rdd.collect()))
    print("This is the partition result: ", union_rdd.getNumPartitions())
    channel.basic_ack(delivery_tag=method.delivery_tag)
