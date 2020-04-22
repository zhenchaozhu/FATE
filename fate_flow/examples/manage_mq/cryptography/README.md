## Usage
1. create 500MB file with
```
dd if=/dev/zero of=testfile bs=1024 count=502400
```

2. generate key pairs

3. use encrypt.py to encrypt data

4. use decrypt.py to decrypt data

5. verify decrpyted file with
```
xdd -p -l100 testfile
```
