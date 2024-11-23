# Commands

### Connect

```bash
ssh [USERNAME]@[HOST] -p [PORT] -i [PRIVATE_KEY_PATH]
```

### SCP

```bash
scp -P [PORT] -i [PRIVATE_KEY_PATH] [SOURCE] [USERNAME]@[HOST]:[DESTINATION]
scp -P [PORT] -i [PRIVATE_KEY_PATH] ./* [USERNAME]@[HOST]:[DESTINATION]
```