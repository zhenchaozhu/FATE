import pyspark, pika
from src import decrypt_blob, encrypt_blob


def HandlePartition(message, public_key, job_id, user, password):

    msg = encrypt_blob(message, public_key)

    credentials = pika.PlainCredentials(user, password)
    connection = pika.BlockingConnection(pika.ConnectionParameters("localhost", 5672, job_id, credentials))
    channel = connection.channel()

    channel.basic_publish(exchange='', routing_key="send-{}".format(job_id), body=msg)

    return [0]
    

sc = pyspark.SparkContext(appName="spark_redis_producer")

fd = open("private_key.pem", "rb")
private_key = fd.read()
fd.close()

fd = open("public_key.pem", "rb")
public_key = fd.read()
fd.close()

lines = []
with open("../tmp_share", "r") as f:
    lines = f.readlines()
job_id = lines[2].replace('\n', "")
user = lines[0].replace('\n', "")
password = lines[1].replace('\n', "")

text_file = sc.textFile("./testfile_50_MB", minPartitions=10)
# counts = text_file.flatMap(lambda line: line.split(" ")) \
#             .map(lambda word: (word, 1)) \
#             .reduceByKey(lambda a, b: a + b)

# bytes_text = text_file.mapPartitions(lambda x: ''.join(x))
encrypt_send_rdd = text_file.mapPartitions(lambda x: HandlePartition(b" ".join(x), public_key, job_id, user, password))


# for method, properties, body in channel.consume(queue="receive-{}".format(job_id)):
#    print("Receive: ", body)
#    channel.basic_ack(delivery_tag=method.delivery_tag)

# decrypt_rdd = encrypt_rdd.mapPartitions(lambda x: decrypt_blob(b''.join(x), private_key))
# encrypt_rdd = text_file.mapPartitions(lambda x: )
# print(text_file.sample(False, 0.1, 81).collect())
# flat_map = text_file.flatMap(lambda line: x.plit)
# print(b''.join(encrypt_rdd.collect()[:1000]))
# print(encrypt_rdd.getNumPartitions())

#print(''.join(text_file.collect()[:1000]))
#print(''.join(decrypt_rdd.collect()[:1000]))
#print(text_file.collect()[:100])
# print(bytes_text.collect()[:10])
print(encrypt_send_rdd.collect())

#print(decrypt_rdd.collect()[:10])
# print(decrypt_rdd.getNumPartitions())
# print(text_file.getNumPartitions())
