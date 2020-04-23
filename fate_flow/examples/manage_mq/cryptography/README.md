## Usage
Please follow "../READMD.md" first to setup the federated queue environment

1. create  files with
```
for i in {1..5}; do $(dd if=/dev/zero of=testfile/file_${i} bs=1024 count=502400); done
```

2. generate key pairs
```
python src/key_generation.py
```

3. submit receive job
```
spark-submit --conf spark.pyspark.python=/usr/bin/python --conf spark.driver.memory=8g --conf spark.executor.memory=8g --master local[2] --py-files src/decrypt.py,src/encrypt.py --conf spark.driver.maxResultSize=0 decrypt_receive.py
```

4. submit send job
```
spark-submit --conf spark.pyspark.python=/usr/bin/python --conf spark.driver.memory=8g --conf spark.executor.memory=8g --master local[2] --py-files src/decrypt.py,src/encrypt.py --conf spark.driver.maxResultSize=0 encrypt_send.py
```

