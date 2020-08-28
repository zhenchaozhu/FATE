#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import shutil
from datetime import datetime

import typing

from fate_flow.manager.model_manager import pipelined_model
from arch.api.utils.core_utils import json_dumps, json_loads
from arch.api.utils.file_utils import get_project_base_directory


def modify_secureboost(model: pipelined_model.PipelinedModel, buffer: dict, src_role: dict, dst_role: dict):
    param_key = None
    for key in buffer:
        if key.endswith("Param"):
            param_key = key

    param_buf = buffer[param_key]
    for i in range(len(param_buf.trees_)):
        for j in range(len(param_buf.trees_[i].tree_)):
            sitename = param_buf.trees_[i].tree_[j].sitename
            role, party_id = sitename.split(":")
            new_party_id = dst_role[role][src_role[role].index(int(party_id))]
            param_buf.trees_[i].tree_[j].sitename = role + ":" + str(new_party_id)

    buffer[param_key] = param_buf
    model.save_component_model('secureboost_0', 'HeteroSecureBoostingTreeGuestParam',
                               'train', buffer)


def migration(config_data: dict):
    require_arguments = ["migrate_initiator", "role", "migrate_role", "model_id", "model_version"]
    check_config(config_data, require_arguments)

    if compare_roles(config_data.get("migrate_role"), config_data.get("role")):
        raise Exception("The config of previous roles is the same with that of migrate roles. "
                        "There is no need to migrate model. Migration process aborting.")

    party_model_id = gen_party_model_id(model_id=config_data["model_id"],
                                        role=config_data["local"]["role"],
                                        party_id=config_data["local"]["party_id"])
    model = pipelined_model.PipelinedModel(model_id=party_model_id,
                                           model_version=config_data["model_version"])
    if not model.exists():
        raise Exception("Can not found {} {} model local cache".format(config_data["model_id"],
                                                                       config_data["model_version"]))
    model_data = model.collect_models(in_bytes=True)
    if "pipeline.pipeline:Pipeline" not in model_data:
        raise Exception("Can not found pipeline file in model.")

    migrate_model = pipelined_model.PipelinedModel(model_id=gen_party_model_id(model_id=gen_model_id(config_data["migrate_role"]),
                                                                               role=config_data["local"]["role"],
                                                                               party_id=config_data["local"]["migrate_party_id"]),
                                                   model_version=config_data["unify_model_version"])

    # migrate_model.create_pipelined_model()
    shutil.copytree(src=model.model_path, dst=migrate_model.model_path)

    pipeline = migrate_model.read_component_model('pipeline', 'pipeline')['Pipeline']

    # Utilize Pipeline_model collect model data. And modify related inner information of model
    train_runtime_conf = json_loads(pipeline.train_runtime_conf)
    train_runtime_conf["role"] = config_data["migrate_role"]
    train_runtime_conf["job_parameters"]["model_id"] = gen_model_id(train_runtime_conf["role"])
    train_runtime_conf["job_parameters"]["model_version"] = migrate_model.model_version
    train_runtime_conf["initiator"] = conf["migrate_initiator"]

    # update pipeline.pb file
    pipeline.train_runtime_conf = json_dumps(train_runtime_conf, byte=True)
    pipeline.model_id = bytes(train_runtime_conf["job_parameters"]["model_id"], "utf-8")
    pipeline.model_version = bytes(train_runtime_conf["job_parameters"]["model_version"], "utf-8")

    # save updated pipeline.pb file
    migrate_model.save_pipeline(pipeline)
    shutil.copyfile(os.path.join(migrate_model.model_path, "pipeline.pb"),
                    os.path.join(migrate_model.model_path, "variables", "data", "pipeline", "pipeline", "Pipeline"))

    secureboost_dict = migrate_model.read_component_model('secureboost_0', 'train')

    modify_secureboost(model=migrate_model, buffer=secureboost_dict, src_role=config_data["role"], dst_role=config_data["migrate_role"])
    print("Migrating model successfully. " \
          "The configuration of model has been modified automatically. " \
          "New model id is: {}, model version is: {}. " \
          "Model files can be found at '{}'.".format(train_runtime_conf["job_parameters"]["model_id"],
                                                     migrate_model.model_version, migrate_model.model_path))


def compare_roles(request_conf_roles: dict, run_time_conf_roles: dict):
    if request_conf_roles.keys() == run_time_conf_roles.keys():
        varify_format = True
        varify_equality = True
        for key in request_conf_roles.keys():
            varify_format = varify_format and (len(request_conf_roles[key]) == len(run_time_conf_roles[key])) and (isinstance(request_conf_roles[key], list))
            request_conf_roles_set = set(str(item) for item in request_conf_roles[key])
            run_time_conf_roles_set = set(str(item) for item in run_time_conf_roles[key])
            varify_equality = varify_equality and (request_conf_roles_set == run_time_conf_roles_set)
        if not varify_format:
            raise Exception("The structure of roles data of local configuration is different from "
                            "model runtime configuration's. Migration aborting.")
        else:
            return varify_equality
    raise Exception("The structure of roles data of local configuration is different from "
                    "model runtime configuration's. Migration aborting.")


def import_from_files(config: dict):
    model = pipelined_model.PipelinedModel(model_id=config["model_id"],
                                           model_version=config["model_version"])
    if config['force']:
        model.force = True
    model.unpack_model(config["file"])


def import_from_db(config: dict):
    model_path = gen_model_file_path(config["model_id"], config["model_version"])
    if config['force']:
        os.rename(model_path, model_path + '_backup_{}'.format(datetime.now().strftime('%Y%m%d%H%M')))


gen_key_string_separator = '#'


def gen_model_file_path(model_id, model_version):
    return os.path.join(get_project_base_directory(), "model_local_cache", model_id, model_version)


def gen_party_model_id(model_id, role, party_id):
    return gen_key_string_separator.join([role, str(party_id), model_id]) if model_id else None


def gen_model_id(all_party):
    return gen_key_string_separator.join([all_party_key(all_party), "model"])


def all_party_key(all_party):
    """
    Join all party as party key
    :param all_party:
        "role": {
            "guest": [9999],
            "host": [10000],
            "arbiter": [10000]
         }
    :return:
    """
    if not all_party:
        all_party_key = 'all'
    elif isinstance(all_party, dict):
        sorted_role_name = sorted(all_party.keys())
        all_party_key = gen_key_string_separator.join([
            ('%s-%s' % (
                role_name,
                '_'.join([str(p) for p in sorted(set(all_party[role_name]))]))
             )
            for role_name in sorted_role_name])
    else:
        all_party_key = None
    return all_party_key


def check_config(config: typing.Dict, required_arguments: typing.List):
    no_arguments = []
    error_arguments = []
    for require_argument in required_arguments:
        if isinstance(require_argument, tuple):
            config_value = config.get(require_argument[0], None)
            if isinstance(require_argument[1], (tuple, list)):
                if config_value not in require_argument[1]:
                    error_arguments.append(require_argument)
            elif config_value != require_argument[1]:
                error_arguments.append(require_argument)
        elif require_argument not in config:
            no_arguments.append(require_argument)
    if no_arguments or error_arguments:
        raise Exception('the following arguments are required: {} {}'.format(','.join(no_arguments), ','.join(['{}={}'.format(a[0], a[1]) for a in error_arguments])))


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], "r") as fin:
        conf = json_loads(fin.read())
    migration(conf)