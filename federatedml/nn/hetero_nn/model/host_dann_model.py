import json
import os

import torch
import torch.nn as nn

from utils import get_latest_timestamp


def create_embedding(size):
    # return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))
    return nn.Embedding(*size, _weight=torch.zeros(*size).normal_(0, 0.01))


def create_embeddings(embedding_meta_dict):
    embedding_dict = dict()
    for key, value in embedding_meta_dict.items():
        embedding_dict[key] = create_embedding(value)
    return embedding_dict


class GlobalModelWrapper(object):
    def __init__(self, classifier, regional_model_list, embedding_dict, partition_data_fn, beta=1.0,
                 pos_class_weight=2.0, loss_name="CE"):
        self.classifier = classifier
        self.regional_model_list = regional_model_list
        self.embedding_dict = embedding_dict
        self.loss_name = loss_name
        if loss_name == "CE":
            self.classifier_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_class_weight]))
        elif loss_name == "BCE":
            self.classifier_criterion = nn.BCEWithLogitsLoss(torch.tensor(pos_class_weight))
        else:
            raise RuntimeError(f"Does not support loss:{loss_name}")
        self.beta = beta
        self.partition_data_fn = partition_data_fn

    def print_parameters(self, print_all=False):
        print("-" * 50)
        print("Global models:")
        for name, param in self.classifier.named_parameters():
            # if param.requires_grad:
            print(f"{name}: {param.data}, {param.requires_grad}")
            # print(f"{name}: {param.requires_grad}")
        if print_all:
            print("Region models:")
            for wrapper in self.regional_model_list:
                wrapper.print_params()
                # for param in wrapper.parameters():
                #     if param.requires_grad:
                #         print(f"{param.train_data}, {param.requires_grad}")
            print("Embedding models:")
            for emb in self.embedding_dict.values():
                for name, param in emb.named_parameters():
                    # if param.requires_grad:
                    # print(f"{name}: {param.data}, {param.requires_grad}")
                    print(f"{name}: {param.requires_grad}")
        print("-" * 50)

    def get_global_classifier_parameters(self, get_tensor=False):
        param_dict = dict()
        for name, param in self.classifier.named_parameters():
            # print("----->", name, param, param.requires_grad)
            if param.requires_grad:
                if get_tensor:
                    param_dict[name] = param
                else:
                    param_dict[name] = param.data.tolist()
        return param_dict

    def load_model(self, root, task_id, task_meta_file_name="task_meta", timestamp=None):
        task_folder = "task_" + task_id
        task_folder_path = os.path.join(root, task_folder)
        if not os.path.exists(task_folder_path):
            raise FileNotFoundError(f"{task_folder_path} is not found.")
        print(f"[INFO] load model from:{task_folder_path}")

        if timestamp is None:
            timestamp = get_latest_timestamp("models_checkpoint", task_folder_path)
            print(f"[INFO] get latest timestamp {timestamp}")

        model_checkpoint_folder = "models_checkpoint_" + str(timestamp)
        model_checkpoint_folder = os.path.join(task_folder_path, model_checkpoint_folder)
        if not os.path.exists(model_checkpoint_folder):
            raise FileNotFoundError(f"{model_checkpoint_folder} is not found.")

        task_meta_file_name = str(task_meta_file_name) + "_" + str(timestamp) + '.json'
        task_meta_file_path = os.path.join(model_checkpoint_folder, task_meta_file_name)
        if not os.path.exists(task_meta_file_path):
            raise FileNotFoundError(f"{task_meta_file_path} is not found.")

        with open(task_meta_file_path) as json_file:
            print(f"[INFO] load task meta file from {task_meta_file_path}")
            task_meta_dict = json.load(json_file)

        # load global classifier
        global_classifier_path = task_meta_dict["global_part"]["classifier"]
        self.classifier.load_state_dict(torch.load(global_classifier_path))
        print(f"[INFO] load global classifier from {global_classifier_path}")

        # load embeddings
        embedding_meta_dict = task_meta_dict["global_part"]["embeddings"]
        for key, emb_path in embedding_meta_dict.items():
            print(f"[INFO] load embedding of [{key}] from {emb_path}")
            self.embedding_dict[key].load_state_dict(torch.load(emb_path))

        # load region models
        region_part_dict = task_meta_dict["region_part"]
        num_region = len(region_part_dict)
        assert num_region == len(self.regional_model_list)

        for idx, region_wrapper in enumerate(self.regional_model_list):
            region = "region_" + str(idx)
            region_wrapper.load_model(region_part_dict[region]["models"])

    def save_model(self, root, task_id, file_name="task_meta", timestamp=None):
        """Save trained model."""

        if timestamp is None:
            raise RuntimeError("timestamp is missing.")

        task_folder = "task_" + task_id
        task_root_folder = os.path.join(root, task_folder)
        if not os.path.exists(task_root_folder):
            os.makedirs(task_root_folder)

        model_checkpoint_folder = "models_checkpoint_" + str(timestamp)
        model_checkpoint_folder = os.path.join(task_root_folder, model_checkpoint_folder)
        if not os.path.exists(model_checkpoint_folder):
            os.makedirs(model_checkpoint_folder)

        extension = ".pth"

        # save global model
        global_classifier = "global_classifier_" + str(timestamp) + extension
        global_classifier_path = os.path.join(model_checkpoint_folder, global_classifier)
        model_meta = dict()
        model_meta["global_part"] = dict()
        model_meta["global_part"]["classifier"] = global_classifier_path
        torch.save(self.classifier.state_dict(), global_classifier_path)
        print(f"[INFO] saved global classifier model to: {global_classifier_path}")

        # save embeddings
        embedding_meta_dict = dict()
        for key, emb in self.embedding_dict.items():
            emb_file_name = "embedding_" + key + "_" + str(timestamp) + extension
            emb_path = os.path.join(model_checkpoint_folder, emb_file_name)
            torch.save(emb.state_dict(), emb_path)
            print(f"[INFO] saved embedding of [{key}] to: {emb_path}")
            embedding_meta_dict[key] = emb_path
        model_meta["global_part"]["embeddings"] = embedding_meta_dict

        # save region models
        model_meta["region_part"] = dict()
        for idx, wrapper in enumerate(self.regional_model_list):
            region = "region_" + str(idx)
            res = wrapper.save_model(model_checkpoint_folder, region + "_" + str(timestamp) + extension)
            model_meta["region_part"][region] = dict()
            model_meta["region_part"][region]["order"] = idx
            model_meta["region_part"][region]["models"] = res

        file_name = str(file_name) + "_" + str(timestamp) + '.json'
        file_full_name = os.path.join(model_checkpoint_folder, file_name)
        with open(file_full_name, 'w') as outfile:
            json.dump(model_meta, outfile)

        return model_meta

    def freeze_top(self, is_freeze=False):
        for param in self.classifier.parameters():
            param.requires_grad = not is_freeze

    def freeze_bottom(self, is_freeze=False, region_idx_list=None):
        # freeze region models
        if region_idx_list is None:
            for wrapper in self.regional_model_list:
                for param in wrapper.parameters():
                    param.requires_grad = not is_freeze
        else:
            print(f"white region idx list:{region_idx_list}")
            for region_idx in region_idx_list:
                for param in self.regional_model_list[region_idx].parameters():
                    param.requires_grad = not is_freeze

        # freeze embedding
        for emb in self.embedding_dict.values():
            for param in emb.parameters():
                param.requires_grad = not is_freeze

    def freeze_bottom_extractors(self, is_freeze=False, region_idx_list=None):
        # freeze region models
        if region_idx_list is None:
            for wrapper in self.regional_model_list:
                for param in wrapper.extractor_parameters():
                    param.requires_grad = not is_freeze
        else:
            print(f"freeze region idx list:{region_idx_list}")
            for region_idx in region_idx_list:
                for param in self.regional_model_list[region_idx].extractor_parameters():
                    param.requires_grad = not is_freeze

    def freeze_bottom_aggregators(self, is_freeze=False, region_idx_list=None):
        # freeze region models
        if region_idx_list is None:
            for wrapper in self.regional_model_list:
                for param in wrapper.aggregator_parameters():
                    param.requires_grad = not is_freeze
        else:
            print(f"freeze region idx list:{region_idx_list}")
            for region_idx in region_idx_list:
                for param in self.regional_model_list[region_idx].aggregator_parameters():
                    param.requires_grad = not is_freeze

    def get_num_regions(self):
        return len(self.regional_model_list)

    def check_discriminator_exists(self):
        for wrapper in self.regional_model_list:
            if wrapper.change_to_train_mode() is False:
                raise RuntimeError('Discriminator not set.')

    def change_to_train_mode(self):
        self.classifier.train()
        for wrapper in self.regional_model_list:
            wrapper.change_to_train_mode()
        for embedding in self.embedding_dict.values():
            embedding.train()

    def change_to_eval_mode(self):
        self.classifier.eval()
        for wrapper in self.regional_model_list:
            wrapper.change_to_eval_mode()
        for embedding in self.embedding_dict.values():
            embedding.eval()

    def parameters(self):
        param_list = list(self.classifier.parameters())
        for wrapper in self.regional_model_list:
            param_list += wrapper.parameters()
        for embedding in self.embedding_dict.values():
            param_list += embedding.parameters()
        return param_list

    def _combine_features(self, feat_dict):
        """

        :param feat_dict:
        :return:
        """

        features_list = []
        embeddings = feat_dict.get("embeddings")
        if embeddings is not None:
            for key, feat in embeddings.items():
                embedding = self.embedding_dict[key]
                feat = feat.long()
                # print(f"key:{key}, feat: \n  {feat}")
                emb = embedding(feat)
                # print(f"key:{key}, emb shape: \n  {emb.shape}")
                features_list.append(emb)
        non_embedding = feat_dict.get("non_embedding")
        if non_embedding is not None:
            # print(f"non_embedding shape:{non_embedding.shape}")
            features_list.append(non_embedding)
        comb_feat_tensor = torch.cat(features_list, dim=1)
        # print(f"comb_feat_tensor shape:{comb_feat_tensor.shape}")
        return comb_feat_tensor

    def forward(self, data, **kwargs):

        feat_dim = data.shape[1] / 2
        source_data, target_data = data[:feat_dim], data[feat_dim:]

        src_wide_list, src_deep_par_list = self.partition_data_fn(source_data)
        tgt_wide_list, tgt_deep_par_list = self.partition_data_fn(target_data)

        # source has domain label of zero, while target has domain label of one
        domain_source_labels = torch.zeros(source_data.shape[0]).long()
        domain_target_labels = torch.ones(target_data.shape[0]).long()

        total_domain_loss = torch.tensor(0.)
        output_list = []
        for wrapper, src_data, tgt_data in zip(self.regional_model_list, src_deep_par_list, tgt_deep_par_list):
            src_feat = self._combine_features(src_data)
            tgt_feat = self._combine_features(tgt_data)
            domain_loss, output = wrapper.compute_total_loss(src_feat, tgt_feat,
                                                             domain_source_labels,
                                                             domain_target_labels,
                                                             **kwargs)
            output_list.append(output)
            total_domain_loss += domain_loss

        # TODO: perform back-propagation based on the total_domain_loss

        output_list = src_wide_list + output_list if len(src_wide_list) > 0 else output_list
        output = torch.cat(output_list, dim=1)
        return output.detach().numpy()

    def backward(self, data, grads):
        data = torch.tensor(data).float()
        grads = torch.tensor(grads).float()
        result = self.predict(data)
        result.backward(gradient=grads)

    def predict(self, data):
        return self._calculate_regional_output(data)

    # def compute_classification_loss(self, data, label):
    #     output = self._calculate_regional_output(data)
    #     pred = self.classifier(output)
    #     if self.loss_name == "CE":
    #         label = label.flatten().long()
    #     else:
    #         # using BCELogitLoss
    #         label = label.reshape(-1, 1).type_as(pred)
    #     class_loss = self.classifier_criterion(pred, label)
    #     return class_loss
    #
    # def calculate_classifier_correctness(self, data, label):
    #     output = self._calculate_regional_output(data)
    #     pred = self.classifier(output)
    #     if self.loss_name == "CE":
    #         # using CrossEntropyLoss
    #         pred_prob = torch.softmax(pred.data, dim=1)
    #         pos_prob = pred_prob[:, 1]
    #         y_pred_tag = pred_prob.max(1)[1]
    #     else:
    #         # using BCELogitLoss
    #         pos_prob = torch.sigmoid(pred.flatten())
    #         y_pred_tag = torch.round(pos_prob).long()
    #
    #     correct_results_sum = y_pred_tag.eq(label).sum().item()
    #     return correct_results_sum, y_pred_tag, pos_prob

    def _calculate_regional_output(self, data):
        wide_list, deep_par_list = self.partition_data_fn(data)
        output_list = []
        if len(deep_par_list) == 0:
            output_list = wide_list
        else:
            for wrapper, data_par in zip(self.regional_model_list, deep_par_list):
                embedding = self._combine_features(data_par)
                output_list.append(wrapper.compute_output(embedding))
            output_list = wide_list + output_list if len(wide_list) > 0 else output_list
        output = torch.cat(output_list, dim=1)
        return output

    def calculate_domain_discriminator_correctness(self, data, is_source=True):
        _, deep_par_list = self.partition_data_fn(data)
        output_list = []
        for wrapper, data_par in zip(self.regional_model_list, deep_par_list):
            embedding = self._combine_features(data_par)
            output_list.append(wrapper.calculate_domain_discriminator_correctness(embedding, is_source=is_source))
        # print(f"[DEBUG] is_source:{is_source}\t domain_discriminator_correctness {output_list}")
        return output_list


class RegionModelWrapper(object):

    def __init__(self, extractor, aggregator, discriminator):
        self.extractor = extractor
        self.aggregator = aggregator
        self.discriminator = discriminator
        self.discriminator_set = False if discriminator is None else True

        # self.classifier_criterion = nn.CrossEntropyLoss()
        self.discriminator_criterion = nn.CrossEntropyLoss()

    def load_model(self, model_dict):
        feature_aggregator_path = model_dict["feature_aggregator"]
        feature_extractor = model_dict["feature_extractor"]
        domain_discriminator = model_dict["domain_discriminator"]
        self.aggregator.load_state_dict(torch.load(feature_aggregator_path))
        self.extractor.load_state_dict(torch.load(feature_extractor))
        self.discriminator.load_state_dict(torch.load(domain_discriminator))
        print(f"[INFO] load aggregator model from {feature_aggregator_path}")
        print(f"[INFO] load extractor model from {feature_extractor}")
        print(f"[INFO] load discriminator model from {domain_discriminator}")

    def save_model(self, model_root, appendix):
        feature_aggregator_name = "feature_aggregator_" + str(appendix)
        feature_extractor_name = "feature_extractor_" + str(appendix)
        domain_discriminator_name = "domain_discriminator_" + str(appendix)
        feature_aggregator_path = os.path.join(model_root, feature_aggregator_name)
        feature_extractor_path = os.path.join(model_root, feature_extractor_name)
        domain_discriminator_path = os.path.join(model_root, domain_discriminator_name)
        torch.save(self.aggregator.state_dict(), feature_aggregator_path)
        torch.save(self.extractor.state_dict(), feature_extractor_path)
        torch.save(self.discriminator.state_dict(), domain_discriminator_path)

        task_meta = dict()
        task_meta["feature_aggregator"] = feature_aggregator_path
        task_meta["feature_extractor"] = feature_extractor_path
        task_meta["domain_discriminator"] = domain_discriminator_path
        print(f"[INFO] saved aggregator model to: {feature_aggregator_path}")
        print(f"[INFO] saved extractor model to: {feature_extractor_path}")
        print(f"[INFO] saved discriminator model to: {domain_discriminator_path}")
        return task_meta

    def check_discriminator_exists(self):
        if self.discriminator_set is False:
            raise RuntimeError('Discriminator not set.')

    def change_to_train_mode(self):
        self.extractor.train()
        self.aggregator.train()
        if self.discriminator_set:
            self.discriminator.train()

    def change_to_eval_mode(self):
        self.extractor.eval()
        self.aggregator.eval()
        if self.discriminator_set:
            self.discriminator.eval()

    def print_params(self):
        print("--" * 50)
        print("==> region classifiers")
        for name, param in self.aggregator.named_parameters():
            # print(f"{name}: {param.data}, {param.requires_grad}")
            print(f"{name}: {param.requires_grad}")
        print("==> region extractors")
        for name, param in self.extractor.named_parameters():
            # print(f"{name}: {param.data}, {param.requires_grad}")
            print(f"{name}: {param.requires_grad}")
        if self.discriminator_set:
            print("==> region discriminators")
            for name, param in self.discriminator.named_parameters():
                # print(f"{name}: {param.data}, {param.requires_grad}")
                print(f"{name}: {param.requires_grad}")

    def parameters(self):
        if self.discriminator_set:
            return list(self.extractor.parameters()) + list(self.aggregator.parameters()) + list(
                self.discriminator.parameters())
        else:
            return list(self.extractor.parameters()) + list(self.aggregator.parameters())

    def extractor_parameters(self):
        return list(self.extractor.parameters())

    def aggregator_parameters(self):
        return list(self.aggregator.parameters())

    def compute_output(self, data):
        batch_feat = self.extractor(data)
        output = self.aggregator(batch_feat)
        return output

    def compute_total_loss(self, source_data, target_data, domain_source_labels, domain_target_labels,
                           **kwargs):
        alpha = kwargs["alpha"]

        num_sample = source_data.shape[0] + target_data.shape[0]
        source_feat = self.extractor(source_data)
        target_feat = self.extractor(target_data)

        output = self.aggregator(source_feat)

        # print("[DEBUG] domain_source_labels should all be zero: \n", domain_source_labels)
        # print("[DEBUG] domain_target_labels should all be one: \n", domain_target_labels)

        domain_feat = torch.cat((source_feat, target_feat), dim=0)
        domain_labels = torch.cat((domain_source_labels, domain_target_labels), dim=0)
        perm = torch.randperm(num_sample)
        domain_feat = domain_feat[perm]
        domain_labels = domain_labels[perm]

        domain_output = self.discriminator(domain_feat, alpha)
        domain_loss = self.discriminator_criterion(domain_output, domain_labels)

        return domain_loss, output

    def calculate_domain_discriminator_correctness(self, data, is_source=True):
        if is_source:
            labels = torch.zeros(data.shape[0]).long()
        else:
            labels = torch.ones(data.shape[0]).long()
        feat = self.extractor(data)
        pred = self.discriminator(feat, alpha=0)
        pred_cls = pred.data.max(dim=1)[1]
        res = pred_cls.eq(labels).sum().item()
        return res
