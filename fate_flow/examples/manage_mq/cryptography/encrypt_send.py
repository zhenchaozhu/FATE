import pyspark, pika
from src import decrypt_blob, encrypt_blob

def HandlePartition(messages, public_key, job_id, user, password):
    for message in messages:
        if len(message) != 0:
            msg = encrypt_blob(message, public_key)

            credentials = pika.PlainCredentials(user, password)
            connection = pika.BlockingConnection(pika.ConnectionParameters("localhost", 5672, job_id, credentials))
            channel = connection.channel()

            channel.basic_publish(exchange='', routing_key="send-{}".format(job_id), body=msg)

    return [0]
    
def FlatMapFunction(element, batch_size=1024*1024*5):
    # default batch sie is 5MB

    print("This is the element", element[:10])

    offset = 0
    break_flag = False

    while not break_flag:
        chunk = element[offset:offset+batch_size]

        if(len(chunk) % batch_size != 0):
            break_flag = True
    
        #print(chunk[:100])
        yield chunk
        offset += batch_size

def FlatMapFunction2(element):
    print("This is the map function", len(element))
    if len(element) != 0:
        yield element

sc = pyspark.SparkContext(appName="spark_redis_producer")

fd = open("private_key.pem", "rb")
private_key = fd.read()
fd.close()

fd = open("public_key.pem", "rb")
public_key = fd.read()
fd.close()

lines = []
# Assume this dir contains manay files
with open("../tmp_share", "r") as f:
    lines = f.readlines()
job_id = lines[2].replace('\n', "")
user = lines[0].replace('\n', "")
password = lines[1].replace('\n', "")

# give it up, just use the default partition counts
# text_file = sc.textFile("./README.md")

# what's the differences between text_file and binary_rdd ?
text_file = sc.textFile("./testfile/*", use_unicode=False)

# need to repartition the rdd
# flat_map = text_file.flatMap(FlatMapFunction2)
encrypt_send_rdd = text_file.mapPartitions(lambda x: HandlePartition(x, public_key, job_id, user, password))

print(encrypt_send_rdd.collect())
# print("This is the flat map partitions: ", flat_map.getNumPartitions())
