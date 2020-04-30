## Usage
1. start containers
```
cd compose
docker-compose up -d
```

2. create federeated queue on guest side
```
python create_federeated_mq_guest.py
```

3. create federeated queue on host side
```
python create_federeated_mq_host.py
```

4. go to sub-dir and run example

for receiver:
```
spark-submit --conf spark.pyspark.python=/usr/bin/python --conf spark.driver.memory=8g --conf spark.executor.memory=8g --master local[2] --py-files cryptography/src/decrypt.py,cryptography/src/encrypt.py cryptography/decrypt_receive.py 
```


for sender:
```
spark-submit --conf spark.pyspark.python=/usr/bin/python --conf spark.driver.memory=8g --conf spark.executor.memory=8g --master local[2] --py-files src/decrypt.py,src/encrypt.py encrypt_send.py 
```

