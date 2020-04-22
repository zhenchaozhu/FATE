import pyspark, pika
from src import decrypt_blob, encrypt_blob

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

    if count == 10:
        print(b''.join(union_rdd.collect()[:10]))
        count = 0
        union_rdd = sc.emptyRDD()

    # if(channel.get_waiting_message_count() == 0):
    #    break
    
    print("Before decrypted: ", body[:10])
    data = decrypt_blob(body, private_key)
    print("After decrypted: ", len(data))

    rdd_test = sc.parallelize(decrypt_blob(body, private_key), 1)
    union_rdd = union_rdd.union(rdd_test)
    channel.basic_ack(delivery_tag=method.delivery_tag)
    count += 1
    # channel.basic_publish(exchange="", routing_key="send-{}".format(job_id), body=json.dumps(test_list))


# for method, properties, body in channel.consume(queue="receive-{}".format(job_id)):
#    print("Receive: ", body)
#    channel.basic_ack(delivery_tag=method.delivery_tag)

# encrypt_rdd = text_file.mapPartitions(lambda x: )
# print(text_file.sample(False, 0.1, 81).collect())
# flat_map = text_file.flatMap(lambda line: x.plit)
# print(b''.join(encrypt_rdd.collect()[:1000]))
# print(encrypt_rdd.getNumPartitions())

#print(''.join(text_file.collect()[:1000]))
#print(''.join(decrypt_rdd.collect()[:1000]))
#print(text_file.collect()[:100])
#print(decrypt_rdd.collect()[:10])
# print(decrypt_rdd.getNumPartitions())
# print(text_file.getNumPartitions())
