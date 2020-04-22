########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

from fate_flow.manager.rabbit_manager import RabbitManager
from fate_flow.utils.tools import RandomNumberString, RandomString
import logging, sys, pika, time

logger = logging.getLogger("")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    rabbit_manager = RabbitManager("guest", "guest", "localhost:15673")
    lines = []
    with open("./tmp_share", "r") as f:
        lines = f.readlines()
    
    # 1. read job ID
    job_id = lines[2].replace('\n', "")
    logger.info("The job id is: %s" % job_id)

    # 2. read username and password
    user = lines[0].replace('\n', "")
    password = lines[1].replace('\n', "")
    logger.info("The username is: %s" % user)
    logger.info("The password is: %s" % password)

    # 3. initial user and vhost
    rabbit_manager.CreateUser(user, password)
    rabbit_manager.CreateVhost(job_id)
    rabbit_manager.AddUserToVhost(user, job_id)

    # 4. initial send queue, the name is send-${vhost}
    send_queue_name = "{}-{}".format("send", job_id)
    rabbit_manager.CreateQueue(job_id, send_queue_name)

    # 5. initial receive queue, the name is receive-${vhost}
    receive_queue_name = "{}-{}".format("receive", job_id)
    rabbit_manager.CreateQueue(job_id, receive_queue_name)

    # replace the ip address with mq broker of the host
    upstream_uri = "amqp://{}:{}@10.193.2.120:5672".format(user, password)
    union_name = rabbit_manager.FederateQueue(upstream_uri ,job_id, receive_queue_name)

    try:
        while True:
            time.sleep(1)

        # 7. test queue
        
    except KeyboardInterrupt:
        # 8. reset every thing
        rabbit_manager.DeFederateQueue(union_name, job_id)
        rabbit_manager.DeleteQueue(job_id, send_queue_name)
        rabbit_manager.DeleteQueue(job_id, receive_queue_name)
        rabbit_manager.DeleteVhost(job_id)
        rabbit_manager.DeleteUser(union_name)
