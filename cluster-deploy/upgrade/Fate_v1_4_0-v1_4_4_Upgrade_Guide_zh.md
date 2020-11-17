# FATE v1.4.4升级包使用说明

**本使用说明仅适用于FATE v1.4.0 - v1.4.4的更新，如有多台机器需要升级，请逐台机器执行此升级脚本。说明中更新包下载路径及解压目录均以用户的home目录为例。**



## 1. 解压升级包

使用如下命令对升级包进行解压:

```bash
cd ~
tar zxvf upgrade_1_4_0-1_4_4.tar.gz
```

解压后将在用户的home目录下得到upgrade_package文件夹。



## 2. 参数修改

进入升级包的解压路径，对解压路径下的`upgrade_script.sh`脚本进行修改，需要修改的参数位于该脚本的5-15行中，以下是对这些参数的解释说明：

```bash
# FATE项目根目录
FATE_ROOT=/data/projects/fate

# FATE FLOW根目录（如果更新项为fatepython或mysql则无需指定flow根目录）
FATE_FLOW_ROOT=/data/projects/fate/python/fate_flow

# 更新包的路径，更新包是位于解压路径下的upgrade_python_package.tar.gz文件
PYTHON_PACKAGE_PATH=~/upgrade_package/upgrade_python_package.tar.gz

# MySQL高权账号（需提供高权账号且通过socket进行数据库连接，否则无法成功更新数据库）
DB_USER=root

# MySQL高权账号的密码
DB_PASS=fate_dev

# MySQL根目录
MYSQL_ROOT=$FATE_ROOT/common/mysql/mysql-8.0.13

# MySQL Sock文件路径
MYSQL_SOCKET_PATH=$MYSQL_ROOT/run/mysql.sock

# 原部署方式（需从allinone和ansible中进行选择修改）
DEPLOY_METHOD=allinone

# 如果部署方式为ansible的情况下，还需指定ansible安装的supervisor根目录
SUPERVISOR_ROOT=/data/projects/common/supervisord
```



## 3. 执行更新脚本

对参数进行修改后，请对更新脚本进行保存，重新打包

```bash
cd ~
tar czf upgrade_package_new.tar.gz upgrade_package
```

### 3.1 allinone部署方式(所有服务部署在同一台机器)

将upgrade_package_new.tar.gz拷贝到部署机器，执行：

```bash
tar xzf upgrade_package_new.tar.gz
cd ./upgrade_package/
sh upgrade_script.sh all
```

如果提示ERROR，... aborting字样，则为参数检查不通过，请根据提示对脚本参数进行二次确认及修改；

如果提示Upgrading process finished字样，则更新成功。


### 3.2 ansible部署方式(所有服务部署在不同的机器)

将upgrade_package_new.tar.gz拷贝到服务部署机器，执行：

```bash
tar xzf upgrade_package_new.tar.gz
cd ./upgrade_package/

# 对应不同的服务执行如下命令
# mysql机器
sh upgrade_script.sh mysql

# fateflow机器
sh upgrade_script.sh fatepython

# nodemanager机器
sh upgrade_script.sh fatepython
因为同一集群不支持多个fateflow，nodemanager机器需要单独停止fateflow服务，参考下面4.1.1

```

如果提示ERROR，... aborting字样，则为参数检查不通过，请根据提示对脚本参数进行二次确认及修改；

### 3.3 升级在线集群连接配置(fateflow所在机器操作)
FATE python代码包在更新时会在同级目录下备份，文件夹名为`python_更新时间`，例如`python_20200901120000`
从备份目录配置文件: ```~/upgrade_package/python_更新日期/fate_flow/settings.py```获取```ZOOKEEPER_HOSTS```
从备份目录配置文件: ```~/upgrade_package/python_更新日期/fate_flow/utils/setting_utils.py```获取```USE_ACL``` ```ZK_USERNAME``` ```ZK_PASSWORD```
编辑/data/projects/fate/python/arch/conf/base_conf.yaml文件, 使用备份配置文件中的值填入新的配置项中
```yaml
use_registry: true
zookeeper:
  hosts:
    - 10.0.0.1:2181
  use_acl: true
  user: fate
  password: fate
```


## 4. 更新回滚

### 4.1 更新所有组件后的回滚操作



#### 4.1.1 停止fate_flow_server服务

#### allinone部署方式的停止服务方法

```bash
# 进入fate_flow组件目录
cd /data/projects/fate/python/fate_flow/

# 停止服务
sh service.sh stop
```

#### ansible部署方式的停止服务方法
```bash
# 进入supervisor组件目录
cd /data/projects/common/supervisord/

# 停止服务
sh service.sh stop fate-fateflow
```



#### 4.1.2 代码更新回退

FATE python代码包在更新时会在同级目录下备份，文件夹名为`python_更新时间`，例如`python_20200901120000`。如需回退，请将更新后的python代码包删除或进行备份，并将原python代码包的名字重新修改为`python`即可。具体操作如下：

```bash
# 进入FATE项目根目录
cd /data/projects/fate/

# 将更新后的python代码包进行备份
mv python/ python_upgrade_backup/

# 将原python代码包重命名为python
mv python_20200901120000/ python/
```



#### 4.1.3 数据库更新回退

```mysql
# 使用高权账号登录MySQL，登录后执行如下操作
# 如果用户使用FATE默认路径安装MySQL，则可以通过如下命令连入数据库：
cd /data/projects/fate/common/mysql/mysql-8.0.13/
./bin/mysql -u root -p -S ./run/mysql.sock

# 选择fate_flow数据库
use fate_flow;

# 对更新过的数据表进行备份
alter table t_queue rename t_queue_backup_20200901;
alter table t_job rename t_job_backup_20200901;
alter table t_task rename t_task_backup_20200901;
alter table t_data_view rename t_data_view_backup_20200901;
alter table t_machine_learning_model_meta rename t_machine_learning_model_meta_backup_20200901;

# 将原数据库表进行恢复
alter table t_queue_backup rename t_queue;
alter table t_job_backup rename t_job;
alter table t_task_backup rename t_task;
alter table t_data_view_backup rename t_data_view;
alter table t_machine_learning_model_meta_backup rename t_machine_learning_model_meta;
```



#### 4.1.4 启动fate_flow_server服务

#### allinone部署方式的启动服务方法

```bash
# 进入fate_flow组件目录
cd /data/projects/fate/python/fate_flow/

# 启动服务
sh service.sh start
```

#### ansible部署方式的停止服务方法

```bash
# 进入supervisor组件目录
cd /data/projects/common/supervisord/

# 启动服务
sh service.sh start fate-fateflow
```



### 4.2 仅更新fatepython组件后的回滚操作

详细操作请见4.1.1、4.1.2、4.1.4步骤。



### 4.3 仅更新mysql组件后的回滚操作

详细操作请见4.1.3步骤。



### 4.4 仅更新fateflow组件后的回滚操作

#### 4.4.1 停止fate_flow_server服务

#### allinone部署方式的停止服务方法

```bash
# 进入fate_flow组件目录
cd /data/projects/fate/python/fate_flow/

# 停止服务
sh service.sh stop
```

#### ansible部署方式的停止服务方法

```bash
# 进入supervisor组件目录
cd /data/projects/common/supervisord/

# 停止服务
sh service.sh stop fate-fateflow
```



#### 4.4.2 代码更新回退

FATE flow代码包在更新时会在同级目录下备份，文件夹名为`fate_flow_更新时间`，例如`fate_flow_20200901120000`。如需回退，请将更新后的fate flow代码包删除或进行备份，并将原fate flow代码包的名字重新修改为`fate_flow`即可。具体操作如下：

```bash
# 进入FATE项目根目录
cd /data/projects/fate/python

# 将更新后的python代码包进行备份
mv fate_flow/ fate_flow_upgrade_backup/

# 将原python代码包重命名为python
mv fate_flow_20200901120000/ fate_flow/
```



#### 4.4.3 启动fate_flow_server服务

#### allinone部署方式的启动服务方法

```bash
# 进入fate_flow组件目录
cd /data/projects/fate/python/fate_flow/

# 启动服务
sh service.sh start
```

#### ansible部署方式的停止服务方法

```bash
# 进入supervisor组件目录
cd /data/projects/common/supervisord/

# 启动服务
sh service.sh start fate-fateflow
```

