## Usage
1. start containers
```
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

4. start consumer process
```
python receive.py
```

5. start producer process
```
python send.py
```