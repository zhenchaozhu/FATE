import json
import os
import time
import sys
import random
from os import path
import subprocess


work_dir = os.path.dirname(__file__)
workflow_engine_dir = path.dirname(path.dirname(__file__))
workflow_home_dir = path.dirname(workflow_engine_dir)
workflow_client = path.join(workflow_home_dir, "python", "fate_flow", "fate_flow_client.py")

def private_subprocess(cmds: list, use_shell: bool = False) -> (str, str):
    # 增加了父环境env识别，避免host版本的python版本异常
    env_private = os.environ.copy()
    env_private.update({"PATH": os.path.dirname(sys.executable), "PYTHONPATH": os.path.realpath(path.join(workflow_home_dir, 'python'))})
    if __debug__:
        print(f"command ->:\n{' '.join(cmds)}")
        subp = subprocess.Popen(cmds,
                                shell=False,
                                stdout=subprocess.PIPE,
                                env=env_private,
                                bufsize=-1)
        # stdout, stderr = subp.communicate()
    else:
        subp = subprocess.Popen(cmds,
                                shell=use_shell,
                                env=env_private,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

    stdout, stderr = subp.communicate()
    stdout = stdout.decode("utf-8")
    if stderr is None:
        stderr = ""
    else:
        stderr = stderr.decode("utf-8")
    return stdout, stderr


def get_time_id():
    return str(int(time.time())) + "_" + str(random.randint(1000, 9999))


def gen_unique_path(prefix):
    return os.path.join(workflow_home_dir, 'data', 'test', prefix + ".config_" + get_time_id())


class TaskManger:
    def __init__(self):
        self.guest_id = 10000
        self.host_id = 9999
        # self.arbiter_id = self.guest_id

        self.work_mode = 0
        self.backend = 0

        self.job_config_path = os.path.join(work_dir, "network_test_conf.json")
        self.job_dsl_path = os.path.join(work_dir, "network_test_dsl.json")

    def make_runtime_conf(self):
        with open(self.job_config_path, 'r', encoding='utf-8') as f:
            json_info = json.loads(f.read())

        json_info['role']['guest'] = [self.guest_id]
        json_info['role']['host'] = [self.host_id]

        json_info['initiator']['party_id'] = self.guest_id
        json_info['job_parameters']['work_mode'] = self.work_mode
        json_info['job_parameters']['backend'] = self.backend

        print(json_info)
        config = json.dumps(json_info)
        config_path = gen_unique_path('submit_job_guest')
        config_dir_path = os.path.dirname(config_path)
        os.makedirs(config_dir_path, exist_ok=True)
        with open(config_path, "w") as f_out:
            f_out.write(config + "\n")
        return config_path

    @staticmethod
    def start_task(cmd):
        stdout, stderr = private_subprocess(cmd)
        try:
            stdout = json.loads(stdout)
        except Exception as e:
            raise RuntimeError(f"start task error, return value: {stdout}, error:{e}")
        return stdout

    def run(self):
        config_dir_path = self.make_runtime_conf()
        start_task_cmd = [sys.executable, workflow_client, "-f", "submit_job", "-c",
                          config_dir_path, "-d", self.job_dsl_path]

        stdout = self.start_task(start_task_cmd)
        status = stdout["retcode"]

        if status != 0:
            raise ValueError(
                "Training task exec fail, status:{}, stdout:{}".format(status, stdout))
        else:
            job_id = stdout["jobId"]

        model_id = stdout['data']['model_info']['model_id']
        model_version = stdout['data']['model_info']['model_version']

        return self.check_loop(job_id)

    def check_loop(self, job_id):
        while True:
            status = self.check_result(job_id)
            if status is None:
                return False

            if status in ["RUNNING", "START", "WAITING"]:
                continue

            if status in ["SUCCESS"]:
                return True

            return False

    def check_result(self, job_id):
        check_cmd = [sys.executable, workflow_client, "-f", "query_job",
                     "-j", job_id, "-r", "guest"]

        stdout = self.start_task(check_cmd)
        try:
            status = stdout["retcode"]
            if status != 0:
                return "RUNNING"
            print("In _check_cpn_status, status: {}".format(status))
            check_data = stdout["data"]
            task_status = check_data[0]['f_status']
            print("Current task status: {}".format(task_status))
            return task_status.upper()
        except:
            return None


def test_network():
    result = TaskManger().run()
    assert result is True


if __name__ == "__main__":
    test_network()
    print("Finished! ")
