## Zookeeper 

 bin/zookeeper-server-start.sh config/zookeeper.properties
 
 ## Kafka
 
bin/kafka-server-start.sh config/server.properties

## Create Topic
bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic sensor


## List Topics

bin/kafka-topics.sh --list --bootstrap-server localhost:9092


## Test a consumer

bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic sensor --from-beginning