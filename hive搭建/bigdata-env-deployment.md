

## 一、集群部署节点和软件版本

### 1.1 集群部署节点规划

| hostname | ip   | memory（g） | cpu  | disk |
| -------- | ---- | ----------- | ---- | ---- |
| bigdata1 |      | 6           | 4    | 50   |
| bigdata2 |      | 6           | 4    | 50   |
| bigdata3 |      | 4           | 2    | 50   |
| bigdata4 |      | 4           | 2    | 50   |
| bigdata5 |      | 4           | 2    | 50   |



| 主机     | NN   | RM   | ZKFC | DN   | NM   | TEZ  | ZK   | JN   | mysql | hive | kafka | phoenix | flink | hbase |
| -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ----- | ------- | ----- | ----- |
| bigdata1 | √    | √    | √    |      |      | √    |      |      |       | √    | √     |         | √     | √     |
| bigdata2 | √    | √    | √    |      |      | √    | √    |      | √     |      | √     |         | √     | √     |
| bigdata3 |      |      |      | √    | √    | √    | √    | √    |       |      |       | √       | √     | √(rs) |
| bigdata4 |      |      |      | √    | √    | √    | √    | √    |       |      |       |         | √     | √(rs) |
| bigdata5 |      |      |      | √    | √    | √    |      | √    |       |      | √     |         | √     | √(rs) |

### 1.2 软件版本

阿里云镜像下软件下载地址：http://mirrors.aliyun.com/apache/

|   软件    | 版本号 |
| :-------: | :----: |
|  hadoop   | 3.1.4  |
|   hive    | 3.1.2  |
|   hbase   | 2.2.6  |
| zookeeper | 3.5.8  |
|   spark   | 2.3.4  |
|   flink   | 1.10.3 |
|   kafka   | 2.4.0  |





## 二、集群搭建

### 2.1 集群初始环境准备

虚拟机安装和系统准备

#### 2.1.1  JDK8 安装

```shell
# 创建安装目录
mkdir java8
# 解压安装包
tar -zxf jdk-8u271-linux-x64.tar.gz
# 配置环境变量
vim ~/.bash_profile
# JAVA_HOME
export JAVA_HOME=/opt/java8/jdk1.8.0_271
export PATH=$PATH:$JAVA_HOME/bin
# SCALA_HOME
export SCALA_HOME=/opt/scala2.11/scala-2.11.12
export PATH=$PATH:$SCALA_HOME/bin

# 验证
source ~/.bash_profile
[root@bigdata5 ~]# java -version
java version "1.8.0_271"
Java(TM) SE Runtime Environment (build 1.8.0_271-b09)
Java HotSpot(TM) 64-Bit Server VM (build 25.271-b09, mixed mode)
```

> scala 环境安装配置和java步骤一样

#### 2.1.2 配置hostname

```shell
vim /etc/hosts
192.168.154.131 bigdata1
192.168.154.132 bigdata2
192.168.154.133 bigdata3
192.168.154.134 bigdata4
192.168.154.135 bigdata5
```

#### 2.1.3 关闭防火墙

```shell
# 查看状态
systemctl status firewalld
# 停止
systemctl stop firewalld
# 禁止开机自启动
systemctl disable firewalld
```

#### 2.1.4 配置时间同步

```shell
# 安装ntp
yum -y install ntp ntpdate
# 查看ntp 转态和启动
systemctl status ntpd
systemctl start ntpd
systemctl enable ntpd
#查看时间和ntp是否开启同步  
date
timedatectl
# 执行timedatectl 命令后，如果 NTP enabled: yes, NTP synchronized: yes，则说明同步时间设置成功。
# 执行timedatectl 命令后，如果 NTP enabled: no 操作如下：
systemctl stop ntpd
timedatectl set-ntp  true
systemctl start  ntpd
# 执行timedatectl 命令后，如果 NTP synchronized: no 操作如下：
systemctl stop ntpd
ntpd -gq   # 重新调整时间  采用平滑的方式同步，重启后稍等一会，synchronized 才会为 true
systemctl start  ntpd
```

![image-20210609001646613](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/ntp-syschronized-no.png)





#### 2.1.5 配置免密码登录

[常用 shell 工具脚本.md](E:\zg-justdoit\bigdata\linux\常用 shell 工具脚本.md)

### 2.2 zookeeper 集群搭建

conf/zoo.cfg 文件

```shell
tickTime=2000
dataDir=/var/lib/zookeeper/data
clientPort=2181
initLimit=10
syncLimit=5
server.2=bigdata2:2888:3888
server.3=bigdata3:2888:3888
server.4=bigdata4:2888:3888
```

需要在每个服务器的 /var/lib/zookeeper/data 下创建 myid 文件，并添加该服务器所对应的 server.n 的n值。
然后分发给其他服务器并修改 myid。
启动zookeeper集群时候需要关闭防火墙，否则不会真正启动服务，查看状态报错: <font color='red'>Error contacting service. It is probably not running .</font>

### 2.3 hadoop 3.1.4 高可用搭建

>官网下载 Hadoop

官网地址：https://hadoop.apache.org/release/3.1.4.html

下载并解压到：/opt/soft 目录下（安装目录自己选择）

> 配置环境变量

```shell
vim ~/.bash_profile

# HADOOP_HOME
export HADOOP_HOME=/opt/soft/hadoop-3.1.4
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

 source ~/.bash_profile
```

> hadoop-env.sh

```shell
export HADOOP_HOME=/opt/soft/hadoop-3.1.4
export JAVA_HOME=/opt/java8/jdk1.8.0_271
export HDFS_NAMENODE_USER=root 
export HDFS_SECONDARYNAMEDODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_JOURNALNODE_USER=root
export HDFS_ZKFC_USER=root
export YARN_NODEMANAGER_USER=root
export YARN_RESOURCEMANAGER_USER=root
export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
```



> core-site.xml

```xml
<configuration>
    <property>
      <name>fs.defaultFS</name>
      <value>hdfs://mycluster</value>
    </property>
    <property>
      <name>hadoop.tmp.dir</name>
      <value>/var/lib/hadoop/tmp</value>
    </property>
	<property>
      <name>dfs.journalnode.edits.dir</name>
      <value>/var/lib/hadoop/journaldata/local/data</value>
    </property>
     <property>
       <name>ha.zookeeper.quorum</name>
       <value>bigdata2:2181,bigdata3:2181,bigdata4:2181</value>
     </property>
</configuration>
```



>hdfs-site.xml

```xml
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>3</value>
    </property>
    <property>
      <name>dfs.nameservices</name>
      <value>mycluster</value>
    </property>
    <property>
      <name>dfs.ha.namenodes.mycluster</name>
      <value>nn1,nn2</value>
    </property>
    <property>
      <name>dfs.namenode.rpc-address.mycluster.nn1</name>
      <value>bigdata1:8020</value>
    </property>
    <property>
      <name>dfs.namenode.rpc-address.mycluster.nn2</name>
      <value>bigdata2:8020</value>
    </property>
    <property>
      <name>dfs.namenode.http-address.mycluster.nn1</name>
      <value>bigdata1:9870</value>
    </property>
    <property>
      <name>dfs.namenode.http-address.mycluster.nn2</name>
      <value>bigdata2:9870</value>
    </property>
    <property>
      <name>dfs.namenode.shared.edits.dir</name>
      <value>qjournal://bigdata3:8485;bigdata4:8485;bigdata5:8485/mycluster</value>
    </property>
    <property>
      <name>dfs.client.failover.proxy.provider.mycluster</name>
      <value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>
    </property>
    <property>
      <name>dfs.ha.fencing.methods</name>
      <value>sshfence</value>
    </property>
    <property>
      <name>dfs.ha.fencing.ssh.private-key-files</name>
      <value>/root/.ssh/id_rsa</value>
    </property>
    <property>
      <name>dfs.ha.fencing.ssh.connect-timeout</name>
      <value>30000</value>
    </property>
    <property>
      <name>dfs.journalnode.edits.dir</name>
      <value>/opt/soft/hadoop-3.1.4/journaldata</value>
    </property>
    <property>
      <name>dfs.ha.nn.not-become-active-in-safemode</name>
      <value>true</value>
    </property>
     <property>
       <name>dfs.ha.automatic-failover.enabled</name>
       <value>true</value>
     </property>
</configuration>
```

>yarn-site.xml

```xml
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.nodemanager.env-whitelist</name>
        <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_MAPRED_HOME</value>
    </property>
    <property>
      <name>yarn.resourcemanager.ha.enabled</name>
      <value>true</value>
    </property>
    <property>
      <name>yarn.resourcemanager.cluster-id</name>
      <value>cluster1</value>
    </property>
    <property>
      <name>yarn.resourcemanager.ha.rm-ids</name>
      <value>rm1,rm2</value>
    </property>
    <property>
      <name>yarn.resourcemanager.hostname.rm1</name>
      <value>bigdata1</value>
    </property>
    <property>
      <name>yarn.resourcemanager.hostname.rm2</name>
      <value>bigdata2</value>
    </property>
    <property>
      <name>yarn.resourcemanager.webapp.address.rm1</name>
      <value>bigdata1:8088</value>
    </property>
    <property>
      <name>yarn.resourcemanager.webapp.address.rm2</name>
      <value>bigdata2:8088</value>
    </property>
    <property>
       <name>yarn.log-aggregation-enable</name>
       <value>true</value>
    </property>
    <property>
      <name>hadoop.zk.address</name>
      <value>bigdata2:2181,bigdata3:2181,bigdata4:2181</value>
    </property>
</configuration>
```

> mapred-site.xml

```xml
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
    <property>
        <name>mapreduce.application.classpath</name>
        <value>$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*</value>
    </property>
    <property>
        <name>yarn.app.mapreduce.am.env</name>
        <value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value>
    </property>
    <property>
        <name>mapreduce.map.env</name>
        <value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value>
    </property>
    <property>
        <name>mapreduce.reduce.env</name>
        <value>HADOOP_MAPRED_HOME=${HADOOP_HOME}</value>
    </property>
    <!-- jobhistory server-->
    <property>
  <name>mapreduce.jobhistory.address</name>
  <value>0.0.0.0:10020</value>
  <description>MapReduce JobHistory Server IPC host:port</description>
</property>

<property>
  <name>mapreduce.jobhistory.webapp.address</name>
  <value>0.0.0.0:19888</value>
  <description>MapReduce JobHistory Server Web UI host:port</description>
</property>

<property>
    <name>mapreduce.jobhistory.done-dir</name>
    <value>/history/done</value>
</property>

<property>
    <name>mapreduce.jobhistory.intermediate-done-dir</name>
    <value>/history/done_intermediate</value>
</property>
</configuration>
```

> workers

```shell
bigdata3
bigdata4
bigdata5
```

> 启动和初始化

1、配置完成后，将配置完成的文件夹 hadoop-3.1.4 远程发送到其他节点当前目录下

```shell
scp -r hadoop-3.1.4/ root@bigdata5:$PWD
```

2、在配置有 journalnode 的节点启动 journalnode

```shell
hdfs --daemon start journalnode
```

3、第一次启动HDFS时，必须对其进行格式化。将一个新的分布式文件系统格式化为hdfs【只需在一个节点上执行一次，比如在bigdata1上执行就行了】

```shell
 hdfs namenode -format mycluster
```

![image-20210610225808729](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/hdfs-namenode-format-successfully.png)

4、在 bigdata1 上启动 namenode

```shell
hdfs --daemon start namenode
```

5、在bigdata2下执行，同步bigdata1 namenode 信息

```shell
hdfs namenode -bootstrapStandby
```

![image-20210610230348150](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/bootstrap-standby-namenode.png)

6、在ZooKeeper中初始化所需的状态。可以通过从其中一个NameNode主机运行以下命令

```shell
# 需要启动 zk 集群
# zkfc 格式化
hdfs zkfc -formatZK
```

将在ZooKeeper中创建一个znode，自动故障转移系统将其数据存储在其中。

![image-20210610230946438](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/zkfc-format-successfully.png)

7、在bigdata 上执行 `stop-all.sh` 停掉 hadoop的所有服务，然后 ` start-all.sh ` 启动服务

![image-20210610231340390](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/start-all-hadoop.png)

![image-20210610232115176](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/hdfs-web.png)

![image-20210610232209093](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/yarn-web.png)

8、启动 JobHistoryServer

```shell
 mapred --daemon start historyserver
```



### 2.4 hbase 集群搭建

下载 hbase 2.2.6 安装包，并解压到 /opt/soft 下。

```shell
tar -zxf hbase-2.2.6-bin.tar.gz
```

> 配置环境变量

```shell
vim ~/.bash_profile

# HBASE_HOME
export HBASE_HOME=/opt/soft/hbase-2.2.6
export PATH=$PATH:$HBASE_HOME/bin

source ~/.bash_profile
```

> 配置 hbase-env.sh

```shell
# 配置 java 路径
export JAVA_HOME=/opt/java8/jdk1.8.0_271
# 不使用 hbase 自己的 zookeeper 管理，设为 false 值来配置使用 自己搭建的zk集群
export HBASE_MANAGES_ZK=false
```

> 配置 hbase-site.xml

```xml
<configuration>
<property>
  <name>hbase.rootdir</name>
  <value>hdfs://mycluster/hbase</value>
</property>
<property>
  <name>hbase.cluster.distributed</name>
  <value>true</value>
</property>
<property>
  <name>hbase.zookeeper.quorum</name>
  <value>bigdata2,bigdata3,bigdata4</value>
</property>
<property>
  <name>hbase.zookeeper.property.dataDir</name>
  <value>/var/lib/zookeeper/data</value>
</property>
</configuration>
```

> 配置 regionservers

```shell
bigdata3
bigdata4
bigdata5
```

> 配置备节点 master，在 conf 下新建文件 *backup-masters*

```shell
bigdata2
```

> 远程拷贝整个hbase 安装目录到其他节点

```shell
[root@bigdata1 soft]# scp -r hbase-2.2.6/ root@bigdata5:$PWD
```

> 启动并测试集群

**首先启动 hdfs 、zookeeper。**

（1）执行 start-hbase.sh 后报错：

```shell
ERROR [master/bigdata1:16000:becomeActiveMaster] master.HMaster: Failed to become active master
java.lang.IllegalStateException: The procedure WAL relies on the ability to hsync for proper operation during component failures, but the underlying
 filesystem does not support doing so. Please check the config value of 'hbase.procedure.store.wal.use.hsync' to set the desired level of robustness
 and ensure the config value of 'hbase.wal.dir' points to a FileSystem mount that can provide it.

```

**解决：**在配置文件 **hbase-site.xml** 中添加一下配置

```xml
  <property>
    <name>hbase.unsafe.stream.capability.enforce</name>
    <value>false</value>
    <description>
      Controls whether HBase will check for stream capabilities (hflush/hsync).

      Disable this if you intend to run on LocalFileSystem, denoted by a rootdir
      with the 'file://' scheme, but be mindful of the NOTE below.

      WARNING: Setting this to false blinds you to potential data loss and
      inconsistent system state in the event of process and/or node failures. If
      HBase is complaining of an inability to use hsync or hflush it's most
      likely not a false positive.
    </description>
  </property>
```

（2）jps 后其他服务器没有 regionserver 和 备用 master 启动成功，查看日志出现如下异常：

```shell
ERROR [main] regionserver.HRegionServer: Failed construction RegionServer
java.lang.IllegalArgumentException: java.net.UnknownHostException: mycluster
        at org.apache.hadoop.security.SecurityUtil.buildTokenService(SecurityUtil.java:417)
```

 **解决：**

a. hbase-env.sh 中添加 HADOOP_CONF_DIR ， 并添加到  HBASE_CLASSPATH 路径中，修改后分发到其他节点；

```shell
HADOOP_CONF_DIR="/opt/soft/hadoop-3.1.4/etc/hadoop"

# Extra Java CLASSPATH elements.  Optional.
# export HBASE_CLASSPATH=
export HBASE_CLASSPATH=$HBASE_CLASSPATH:$HADOOP_CONF_DIR
```

b. 复制 *hdfs-site.xml*  或者 链接 到 *${HBASE_HOME}/conf* 。

![image-20210613181629717](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/hbase集群启动成功进程.png)

![image-20210613181800075](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/hbase的web ui.png)



### 2.5 mysql 5.7 安装

> 下载mysql源安装包

```shell
wget http://dev.mysql.com/get/mysql57-community-release-el7-8.noarch.rpm
```

> 安装mysql源

```shell
yum -y install mysql57-community-release-el7-8.noarch.rpm
```

> 检查mysql源是否安装成功

```shell
yum repolist enabled | grep "mysql.*-community.*"
```

> 安装MySQL

```shell
yum -y install mysql-community-server
```

> 启动MySQL服务

```shell
systemctl start mysqld
```

> 查看MySQL的启动状态

```shell
systemctl status mysqld
```

> 开机启动

```shell
systemctl enable mysqld 
systemctl daemon-reload
```

> 修改root本地登录密码

mysql安装完成之后，在/var/log/mysqld.log文件中给root生成了一个默认密码。

 ```shell
 grep 'temporary password' /var/log/mysqld.log # 查看默认生成的密码
 mysql -uroot -p
 # 密码设置复杂些
 ALTER USER 'root'@'localhost' IDENTIFIED BY 'Root@0625';
 或
 set password for 'root'@'localhost'=password('Root@0625');
 ```

如果密码设置简单，不满足密码安全检查，则报错：
<font color='red'>ERROR 1819 (HY000): Your password does not satisfy the current policy requirements</font>

mysql5.7默认安装了密码安全检查插件(validate_password)，默认密码检查策略要求密码必须包含：大小写字母、数字和特殊符号，并且长度不能少于8位。

MySQL官网密码策略详细说明：https://dev.mysql.com/doc/refman/5.7/en/validate-password-options-variables.html#sysvar_validate_password_policy

> 添加远程登录用户

默认只允许root帐户在本地登录，如果要在其它机器上连接mysql，必须修改root允许远程连接，或者添加一个允许远程连接的帐户，为了安全起见，我添加一个新的帐户：

```sql
GRANT ALL PRIVILEGES ON *.* TO 'zgjustdoit'@'%' IDENTIFIED BY 'ZGjustdoit@0430' WITH GRANT OPTION;
---- root 用户
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY 'Root@0625' WITH GRANT OPTION;
```

> 配置默认编码为utf8

修改/etc/my.cnf配置文件，在[mysqld]下添加编码配置：

```shell
[mysqld]
character_set_server=utf8
init_connect='SET NAMES utf8'
```

重启mysql，查看变量：

```sql
show variables like '%character%';
```



### 2.6 hive 安装

> hive 元数据存 mysql 关系型数据库，所以先确保安装了mysql。hive 安装在 bigdata1 上。

mysql 中创建 hive 数据库，字符集：utf8；排序规则：utf8_general_ci。（直接使用 Navicat 创建）



> 官网下载，解压

```shell
# 切换到 /opt/soft 目录
cd /opt/soft
# 下载安装包
wget https://mirrors.bfsu.edu.cn/apache/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz
# 解压
tar zxf apache-hive-3.1.2-bin.tar.gz
```

> 配置环境变量

```shell
vim ~/.bash_profile
# HIVE_HOME
export HIVE_HOME=/opt/soft/hive-3.1.2
export PATH=$PATH:$HIVE_HOME/bin

source ~/.bash_profile
```

> HDFS 中创建目录，并赋予读写权限

```shell
hdfs dfs -mkdir -p /user/hive/warehouse
hdfs dfs -mkdir -p /user/hive/tmp
hdfs dfs -chmod 777 /user/hive/warehouse
hdfs dfs -chmod 777 /user/hive/tmp

hdfs dfs -ls  /user/hive
Found 2 items
drwxrwxrwx   - root supergroup          0 2021-06-14 00:26 /user/hive/tmp
drwxrwxrwx   - root supergroup          0 2021-06-14 00:26 /user/hive/warehouse
```

> 配置 hive-env.sh

```shell

# Set HADOOP_HOME to point to a specific hadoop install directory
# HADOOP_HOME=${bin}/../../hadoop
HADOOP_HOME=${HADOOP_HOME}

# Hive Configuration Directory can be controlled by:
# export HIVE_CONF_DIR=
export HIVE_CONF_DIR=/opt/soft/hive-3.1.2/conf

# Folder containing extra libraries required for hive compilation/execution can be controlled by:
# export HIVE_AUX_JARS_PATH=
export HIVE_AUX_JARS_PATH=/opt/soft/hive-3.1.2/lib

```

> 配置 log 输出路径 hive-log4j2.properties

在 hive 家目录下创建 logs。

```shell
mkdir logs

vim  hive-log4j2.properties
property.hive.log.dir = /opt/soft/hive-3.1.2/logs

```

> 配置 hive-site.xml

```shell
 cp hive-default.xml.template hive-site.xml
```

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?><!--
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
--><configuration>
  <!-- WARNING!!! This file is auto generated for documentation purposes ONLY! -->
  <!-- WARNING!!! Any changes you make to this file will be ignored by Hive.   -->
  <!-- WARNING!!! You must make your changes in hive-site.xml instead.         -->
  <!-- Hive Execution Parameters -->
  
  <property>
    <name>hive.exec.scratchdir</name>
    <value>/user/hive/tmp</value>
    <description>HDFS root scratch dir for Hive jobs which gets created with write all (733) permission. For each connecting user, an HDFS scratch dir: ${hive.exec.scratchdir}/&lt;username&gt; is created, with ${hive.scratch.dir.permission}.</description>
  </property>
    <property>
    <name>hive.metastore.warehouse.dir</name>
    <value>/user/hive/warehouse</value>
    <description>location of default database for the warehouse</description>
  </property>
    <property>
    <name>hive.metastore.uris</name>
    <value>thrift://bigdata1:9083</value>
    <description>Thrift URI for the remote metastore. Used by metastore client to connect to remote metastore.</description>
  </property>
    <property>
    <name>hive.server2.thrift.bind.host</name>
    <value>bigdata1</value>
    <description>Bind host on which to run the HiveServer2 Thrift service.</description>
  </property>
  <property>
    <name>hive.server2.thrift.port</name>
    <value>10000</value>
    <description>Port number of HiveServer2 Thrift interface when hive.server2.transport.mode is 'binary'.</description>
  </property>
  <property>
    <name>hive.metastore.event.db.notification.api.auth</name>
    <value>false</value>
    <description>
      Should metastore do authorization against database notification related APIs such as get_next_notification.
      If set to true, then only the superusers in proxy settings have the permission
    </description>
  </property>
  <property>
    <name>hive.server2.active.passive.ha.enable</name>
    <value>true</value>
    <description>Whether HiveServer2 Active/Passive High Availability be enabled when Hive Interactive sessions are enabled.This will also require hive.server2.support.dynamic.service.discovery to be enabled.</description>
  </property>
  <property>
    <name>javax.jdo.option.ConnectionDriverName</name>
    <value>com.mysql.cj.jdbc.Driver</value>
    <description>Driver class name for a JDBC metastore</description>
  </property>
  <property>
    <name>javax.jdo.option.ConnectionURL</name>
    <value>jdbc:mysql://bigdata2:3306/hive?createDatabaseIfNotExist=true&amp;useSSL=false</value>
    <description>
      JDBC connect string for a JDBC metastore.
      To use SSL to encrypt/authenticate the connection, provide database-specific SSL flag in the connection URL.
      For example, jdbc:postgresql://myhost/db?ssl=true for postgres database.
    </description>
  </property>
  <property>
    <name>javax.jdo.option.ConnectionUserName</name>
    <value>root</value>
    <description>Username to use against metastore database</description>
  </property>  
  <property>
    <name>javax.jdo.option.ConnectionPassword</name>
    <value>Root@0625</value>
    <description>password to use against metastore database</description>
  </property>  
  
</configuration>

```

> 补充 jar 包

添加 mysql-connector-java-6.0.6.jar 到 hive 的 lib 目录下。

升级 guava 的jar包，否则在执行下面的 schematool 工具初始化时候报异常。将原来的 guava-19.0.jar 替换为下载好的  guava-29.0-jre.jar。

> 启动

```shell
# 1. Hive 2.1 版本开始，需要执行 schematool  命令初始化元数据
schematool -dbType mysql -initSchema -verbose
```

![image-20210614011630780](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/schematoop-init-completed.png)



```shell
# 2.启动metastore和hiveserver2
nohup hive --service metastore --hiveconf hive.log.file=metastore.log &>> /opt/soft/hive-3.1.2/logs/metastore.log &
nohup hive --service hiveserver2 --hiveconf hive.log.file=hiveserver2.log &>> /opt/soft/hive-3.1.2/logs/hiveserver2.log &
```

![image-20210614012355740](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/start-metastore-hiveserver2-successfully.png)

```shell
# 3. 执行
beeline -u jdbc:hive2://bigdata1:10000  
```

不能进入 beeline， 查看 hiveserver2 日志发现异常：

<font color='red'>Caused by: java.lang.RuntimeException: java.lang.RuntimeException: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.security.authorize.AuthorizationException): User: root is not allowed to impersonate anonymous.</font>

**解决：**

修改hadoop 配置文件 etc/hadoop/core-site.xml,加入如下配置项，修改后分发到每个节点，重启hadoop集群：

*root*  指的是报错信息User 后的用户名。

```xml
<property>
    <name>hadoop.proxyuser.root.hosts</name>
    <value>*</value>
</property>
<property>
    <name>hadoop.proxyuser.root.groups</name>
    <value>*</value>
</property>
```

再次执行，成功：

![image-20210614013946270](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/beeline-successful.png)



### 2.7 kafka 集群搭建

>下载并解压安装包

```shell
wget http://archive.apache.org/dist/kafka/2.4.0/kafka_2.11-2.4.0.tgz
tar zxf kafka_2.11-2.4.0.tgz
```

> 配置  server.properties

```properties
# The id of the broker. This must be set to a unique integer for each broker.
# 每个节点需要不一样
broker.id=1  
# A comma separated list of directories under which to store log files
log.dirs=/var/lib/kafka/kafka-logs
# The minimum age of a log file to be eligible for deletion due to age
# 只保留 3 天
log.retention.hours=72
# 配置使用自己搭建的zookeeper，没有配置kafka 自身的zookeeper
zookeeper.connect=bigdata2:2181,bigdata3:2181,bigdata4:2181
```



> 配置环境变量

```shell
vim ~/.bash_profile
# KAFKA_HOME
export KAFKA_HOME=/opt/soft/kafka_2.11-2.4.0
export PATH=$PATH:$KAFKA_HOME/bin
 
source ~/.bash_profile
```



> 每个节点上启动kafka server

需要先启动 zookeeper 集群。

```shell
[root@bigdata1 scripts]# cat kafkaOps.sh
#!/bin/bash

base_path="${KAFKA_HOME}"
base_server_sh="${base_path}/bin"
base_server_config="${base_path}/config/server.properties"
# all node names
NODES=()
###### Validation args length
if [[ $# -lt 1 ]];then
  echo "Exit: At least one parameters, eg: start|stop|status"
  exit 2
fi

###### Get all node name
hostnames="bigdata1 bigdata2 bigdata5"
i=0
for node in $hostnames
do
   NODES[i]=$node
   let i++
done

case $1 in
"start"){
    for node in ${NODES[@]}
        do
          # 后台启动
            ssh ${node} "source ~/.bash_profile; ${base_server_sh}/kafka-server-start.sh -daemon ${base_server_config}"
            echo "${node} start kafka";
        done
};;
"stop"){
    for node in ${NODES[@]}
        do
            ssh ${node} "source ~/.bash_profile; ${base_server_sh}/kafka-server-stop.sh ${base_server_config}"
            echo "${node} stop kafka";
        done
};;
esac
```

> 创建 topic

```shell
kafka-topics.sh --bootstrap-server bigdata2:9092 --replication-factor 2 --partitions 3 --create  --topic zgtest
kafka-topics.sh --bootstrap-server bigdata2:9092 --list
# zgtest
```

> kafka shell 实现发送和消费消息

producer:

```shell
kafka-console-producer.sh --broker-list bigdata1:9092 --topic zgtest
```

![image-20210614123708371](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/kafka-console-producer-zgtest.png)

consumer:

```shell
kafka-console-consumer.sh --bootstrap-server bigdata2:9092 --from-beginning  --topic zgtest
```

![image-20210614123635512](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/kafka-console-consumer.png)



### 2.8 flink on yarn 集群搭建

> 下载解压，配置环境变量

```shell
# FLINK_HOME
export FLINK_HOME=/opt/soft/flink-1.10.3/
export PATH=$PATH:$FLINK_HOME/bin
```

> 配置 masters

```shell
bigdata1:8081
bigdata2:8081
```

> 配置 slaves

```shell
bigdata3
bigdata4
bigdata5
```

> 配置 conf/flink-conf.yaml

```yaml
# The high-availability mode. Possible options are 'NONE' or 'zookeeper'.
#
high-availability: zookeeper
high-availability.storageDir: hdfs://mycluster/flink/ha/
high-availability.zookeeper.quorum: bigdata2:2181,bigdata3:2181,bigdata4:2181
high-availability.zookeeper.path.root: /flink-yarn
yarn.application-attempts: 10
```



> 添加 hadoop 与 flink 集成的jar 包到flink lib目录下

[flink-shaded-hadoop-2-uber-2.8.3-10.0.jar](https://repo.maven.apache.org/maven2/org/apache/flink/flink-shaded-hadoop-2-uber/2.8.3-10.0/flink-shaded-hadoop-2-uber-2.8.3-10.0.jar) 

> 配置 start-cluster.sh

配置启动脚本添加 hadoop 配置路径：

```shell
export HADOOP_CONF_DIR="/opt/soft/hadoop-3.1.4/etc/hadoop"
```



### 2.9 tez 安装及hive on tez 配置

启动hive，出现如下日志：

```shell
Hive-on-MR is deprecated in Hive 2 and may not be available in the future versions. Consider using a different execution engine (i.e. spark, tez) or using Hive 1.X releases.
```

> 下载并解压 tez

官网下载安装包：[apache-tez-0.10.0-bin.tar.gz](https://mirrors.bfsu.edu.cn/apache/tez/0.10.0/apache-tez-0.10.0-bin.tar.gz)

```shell
tar -zxf apache-tez-0.10.0-bin.tar.gz
mv apache-tez-0.10.0-bin tez-0.10.0
```

> 将tez的压缩包put到hdfs上去

```shell
hdfs dfs -mkdir -p /user/tez
hdfs dfs -put /opt/soft/tez-0.10.0/share/tez.tar.gz   /user/tez
hdfs dfs -ls /user/tez
```

> 在 hadoop-3.1.4/etc/hadoop/ 下创建tez-site.xml文件并写上如下配置

需要将该文件发送到所有节点

```xml
<configuration>                                                              
  <property>  
     <name>tez.lib.uris</name>  
     <value>${fs.defaultFS}/user/tez/tez.tar.gz</value>  <!-- 这里指向hdfs上的tez.tar.gz包 -->  
  </property>  
  <property>  
     <name>tez.container.max.java.heap.fraction</name>  <!-- 设置 task/AM 占用JVM内存大小的比例，默认值是0.8，内存不足时一般可以调小 -->  
     <value>0.4</value>  
  </property>  
</configuration>
```

> 配置环境变量，每个节点都配置

```shell
 vim ~/.bash_profile

#TEZ_HOME
export TEZ_HOME=/opt/soft/tez-0.10.0
export TEZ_CONF_DIR=${HADOOP_HOME}/etc/hadoop/tez-site.xml
export TEZ_JARS=$TEZ_HOME/*:$TEZ_HOME/lib/*
export HADOOP_CLASSPATH=$TEZ_CONF_DIR:$TEZ_JARS:$HADOOP_CLASSPATH

source  ~/.bash_profile
```

> 与hadoop的兼容问题

TEZ 安装目录lib下的hadoop相关jar版本：

![image-20210714154059214](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/TEZ 安装目录lib下的hadoop相关jar版本.png)

安装的hadoop版本是3.1.4，需要将hadoop share目录下的对应的jar包覆盖Tez 家目录lib下：

```shell
cp $HADOOP_HOME/share/hadoop/hdfs/hadoop-hdfs-client-3.1.4.jar /opt/soft/tez-0.10.0/lib
cp $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-client-common-3.1.4.jar /opt/soft/tez-0.10.0/lib
cp $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-client-core-3.1.4.jar /opt/soft/tez-0.10.0/lib
cp $HADOOP_HOME/share/hadoop/yarn/hadoop-yarn-server-timeline-pluginstorage-3.1.4.jar /opt/soft/tez-0.10.0/lib
rm -f /opt/soft/tez-0.10.0/lib/hadoop-hdfs-client-3.1.3.jar
rm -f /opt/soft/tez-0.10.0/lib/hadoop-mapreduce-client-common-3.1.3.jar
rm -f /opt/soft/tez-0.10.0/lib/hadoop-mapreduce-client-core-3.1.3.jar
rm -f /opt/soft/tez-0.10.0/lib/hadoop-yarn-server-timeline-pluginstorage-3.1.3.jar
```

> 拷贝Tez配置好的安装包到每个节点

```shell
scp -r /opt/soft/tez-0.10.0/ root@bigdata2:$PWD
```

> 重启hadoop

（1）hive 或 beeline 连接进入到hive中，切换执行引擎

```shell
# tez
set hive.execution.engine=tez;
# mr
set hive.execution.engine=mr;
# spark
set hive.execution.engine=spark;
```

(2) 修改hive 配置文件

可以在hive的conf目录下修改hive-site.xml， 修改之后重启metastore和hiveserver2，再执行操作就会默认执行引擎为Tez。

```shell
vim hive-site.xml

<property>
	<name>hive.execution.engine</name>
	<value>tez</value>
</property>
```

> 配置 tez-ui

(1) 下载安装tomcat
[官网下载地址](https://tomcat.apache.org/download-80.cgi)

![image-20210714172124892](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/tomcat下载.png)



可以在conf目录下server.xml中修改端口号，修改为了8888：
![image-20210714191838817](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/tomcat server端口号修改.png)

启动 tomcat：

```shell
#在bin目录下运行startup.sh
sh startup.sh
```

访问http://bigdata1:8889/看到tomcat的欢迎页面。bigdata1是我虚拟机主机名。
![image-20210714192304053](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/tomcat启动成功欢迎页.png)



（2） 下载 tez-ui war包并部署在 tomcat

[tez-ui-0.10.0.war](https://repository.apache.org/content/repositories/releases/org/apache/tez/tez-ui/0.10.0/tez-ui-0.10.0.war)

```shell
# 在 tomcat的 webapp下创建文件夹 tez-ui
 mkdir tez-ui
# 将 tez-ui的war包传到该文件夹下，并解压，修改配置文件
unzip tez-ui-0.10.0.war
vim configs.env
```

![image-20210714193155520](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/tez-ui 配置文件修改.png)

（3）配置timelineserver

**vim yarn-site.xml添加以下内容.然后分发到各个节点：**

```xml

<!-- conf timeline server -->
<property>
     <name>yarn.timeline-service.enabled</name>
     <value>true</value>
</property>
<property>
     <name>yarn.timeline-service.hostname</name>
     <value>bigdata1</value>
</property>
<property>
     <name>yarn.timeline-service.http-cross-origin.enabled</name>
     <value>true</value>
</property>
<property>
     <name> yarn.resourcemanager.system-metrics-publisher.enabled</name>
     <value>true</value>
</property>
<property>
     <name>yarn.timeline-service.generic-application-history.enabled</name>
     <value>true</value>
</property>
<property>
     <description>Address for the Timeline server to start the RPC server.</description>
     <name>yarn.timeline-service.address</name>
     <value>bigdata1:10201</value>
</property>
<property>
     <description>The http address of the Timeline service web application.</description>
     <name>yarn.timeline-service.webapp.address</name>
     <value>bigdata1:8188</value>
</property>
<property>
     <description>The https address of the Timeline service web application.</description>
     <name>yarn.timeline-service.webapp.https.address</name>
     <value>bigdata1:2191</value>
</property>
<property>
     <name>yarn.timeline-service.handler-thread-count</name>
     <value>12</value>
</property>
```

**vim tez-site.xml添加下列几行，分发到各个节点:**

```xml
<!--Configuring Tez to use YARN Timeline-->
<property>
    <name>tez.history.logging.service.class</name>
    <value>org.apache.tez.dag.history.logging.ats.ATSHistoryLoggingService</value>
</property>
<property>
    <name>tez.tez-ui.history-url.base</name>
    <value>http://bigdata1:8889/tez-ui/</value>
</property>

```

（4）重启hadoop

```shell
stop-all.sh
start-all.sh
```

（5）启动timelineserver

```shell
yarn --daemon start timelineserver
```

（6）启动 tomcat

```shell
/opt/soft/tomcat8.5.69/bin/startup.sh
```

![image-20210714211738521](https://gitee.com/zg-justdoit/image/raw/master/markdown-image/tez-ui访问界面.png)























































































