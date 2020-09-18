import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


def adjust_learning_rate(original_lr, **kwargs):
    epochs = kwargs["max_epochs"]
    num_batch = kwargs["num_batch"]
    curr_epoch = kwargs["current_epoch"]
    batch_idx = kwargs["batch_idx"]
    start_steps = curr_epoch * num_batch
    total_steps = epochs * num_batch
    p = float(batch_idx + start_steps) / total_steps

    beta = 0.75
    alpha = 10
    lr = original_lr / (1 + alpha * p) ** beta
    return lr, p


def create_embedding(size):
    return nn.Embedding(*size, _weight=torch.zeros(*size).normal_(0, 0.01))


def create_embeddings(embedding_meta_dict):
    embedding_dict = dict()
    for key, value in embedding_meta_dict.items():
        embedding_dict[key] = create_embedding(value)
    return embedding_dict


class HostDannModel(object):
    def __init__(self, regional_model_list, embedding_dict, partition_data_fn, optimizer_param, beta=1.0,
                 pos_class_weight=2.0, loss_name="CE"):
        self.classifier = None
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
        self._init_optimizer(optimizer_param)

    def _init_optimizer(self, optimizer_param):
        self.original_learning_rate = optimizer_param.kwargs["learning_rate"]
        self.classifier_learning_rate = optimizer_param.kwargs["classifier_learning_rate"]

    def print_parameters(self):
        print("-" * 50)
        print("Region models:")
        for wrapper in self.regional_model_list:
            wrapper.print_parameters()
            # for param in wrapper.parameters():
            #     if param.requires_grad:
            #         print(f"{param.train_data}, {param.requires_grad}")
        if self.embedding_dict is not None:
            print("Embedding models:")
            for emb in self.embedding_dict.values():
                for name, param in emb.named_parameters():
                    # if param.requires_grad:
                    # print(f"{name}: {param.data}, {param.requires_grad}")
                    print(f"{name}: {param.requires_grad}")
        print("-" * 50)

    def state_dict(self):
        # save embeddings
        model_state_dict = dict()

        if self.embedding_dict is not None:
            embeddings_state_dict = dict()
            for key, emb in self.embedding_dict.items():
                embeddings_state_dict[key] = emb.state_dict()
            model_state_dict["embeddings"] = embeddings_state_dict

        # save region models
        model_state_dict["regional_models"] = dict()
        for idx, regional_model in enumerate(self.regional_model_list):
            region = "region_" + str(idx)
            model_state_dict["regional_models"][region] = dict()
            model_state_dict["regional_models"][region]["order"] = idx
            model_state_dict["regional_models"][region]["models"] = regional_model.state_dict()
        return model_state_dict

    def load_state_dict(self, model_state_dict):
        # load embeddings
        if self.embedding_dict is not None:
            embeddings_state_dict = model_state_dict["embeddings"]
            for key, emb_state_dict in embeddings_state_dict.items():
                self.embedding_dict[key].load_state_dict(emb_state_dict)

        # load region models
        regional_models_dict = model_state_dict["regional_models"]
        num_region = len(regional_models_dict)
        assert num_region == len(self.regional_model_list)

        for idx, regional_model in enumerate(self.regional_model_list):
            region = "region_" + str(idx)
            regional_model.load_state_dict(regional_models_dict[region]["models"])

    def export_model(self):
        f = tempfile.TemporaryFile()
        try:
            torch.save(self.state_dict(), f)
            f.seek(0)
            model_bytes = f.read()
            return model_bytes
        finally:
            f.close()

    def restore_model(self, model_bytes):
        f = tempfile.TemporaryFile()
        f.write(model_bytes)
        f.seek(0)
        self.load_state_dict(torch.load(f))
        f.close()

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

        if self.embedding_dict is not None:
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
        for wrapper in self.regional_model_list:
            wrapper.change_to_train_mode()
        if self.embedding_dict is not None:
            for embedding in self.embedding_dict.values():
                embedding.train()

    def change_to_eval_mode(self):
        for wrapper in self.regional_model_list:
            wrapper.change_to_eval_mode()
        if self.embedding_dict is not None:
            for embedding in self.embedding_dict.values():
                embedding.eval()

    def parameters(self):
        param_list = list()
        for regional_model in self.regional_model_list:
            param_list += regional_model.parameters()
        if self.embedding_dict is not None:
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
            if self.embedding_dict is None:
                raise Exception("No embedding model is provided.")

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

    def forward(self, x, **kwargs):
        output = self._forward(x, **kwargs)
        return output.detach().numpy()

    def local_train(self, x, y, **kwargs):
        if self.classifier is None:
            raise RuntimeError("The label classifier is not specified.")

        parameters = self.parameters() + list(self.classifier.parameters())
        optimizer = optim.SGD(parameters, lr=self.classifier_learning_rate, momentum=0.9, weight_decay=0.01)

        output = self._forward(x, **kwargs)
        prediction = self.classifier(output)

        if self.loss_name == "CE":
            y = y.flatten().long()
        else:
            # using BCELogitLoss
            y = y.reshape(-1, 1).type_as(prediction)

        class_loss = self.classifier_criterion(prediction, y)
        class_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _forward(self, x, **kwargs):
        msg = f"[DEBUG] HostDannModel.forward"
        LOGGER.debug(msg)
        print(msg)
        x = torch.tensor(x).float()

        feat_dim = int(x.shape[1] / 2)
        msg = f"[DEBUG] source and target feat_dim:{feat_dim}"
        LOGGER.debug(msg)
        print(msg)

        # TODO: split the data into source data and target data
        # TODO: This is temporary and it may change
        source_data, target_data = x[:, :feat_dim], x[:, feat_dim:]
        msg = f"[DEBUG] source.shape and target.shape feat_dim:{source_data.shape}, {target_data.shape}"
        LOGGER.debug(msg)

        src_wide_list, src_deep_par_list = self.partition_data_fn(source_data)
        tgt_wide_list, tgt_deep_par_list = self.partition_data_fn(target_data)

        # source has domain label of zero, while target has domain label of one
        domain_source_labels = torch.zeros(source_data.shape[0]).long()
        domain_target_labels = torch.ones(target_data.shape[0]).long()

        msg = f"[DEBUG] lengths: {len(self.regional_model_list)}, {len(src_deep_par_list)}, {len(tgt_deep_par_list)}"
        LOGGER.debug(msg)

        curr_lr, p = adjust_learning_rate(original_lr=self.original_learning_rate, **kwargs)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        kwargs["alpha"] = alpha

        LOGGER.info(f"dann curr_lr:{curr_lr}, p:{p}, alpha:{alpha}")

        total_domain_loss = torch.tensor(0.)
        output_list = []
        for regional_model, src_data, tgt_data in zip(self.regional_model_list, src_deep_par_list, tgt_deep_par_list):
            src_feat = self._combine_features(src_data)
            tgt_feat = self._combine_features(tgt_data)
            # msg = f"[DEBUG] src_feat:{src_feat}, tgt_feat:{tgt_feat}"
            # LOGGER.debug(msg)
            domain_loss, output = regional_model.compute_total_loss(src_feat, tgt_feat,
                                                                    domain_source_labels,
                                                                    domain_target_labels,
                                                                    **kwargs)
            output_list.append(output)
            total_domain_loss += domain_loss

        self._train_regional_models(total_domain_loss, curr_lr)

        output_list = src_wide_list + output_list if len(src_wide_list) > 0 else output_list
        return torch.cat(output_list, dim=1)

    def _train_regional_models(self, loss, curr_lr):
        msg = f"[DEBUG] train_regional_models with total_domain_loss:{loss} and learning rate:{curr_lr}"
        LOGGER.debug(msg)

        # self.print_parameters()
        optimizer = optim.SGD(self.parameters(), lr=curr_lr, momentum=0.9)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        msg = f"[DEBUG] finished HostDannModel.train_local_model"
        LOGGER.debug(msg)

    def backward(self, x, grads, **kwargs):
        msg = f"[DEBUG] HostDannModel.backward"
        LOGGER.debug(msg)
        print(msg)

        x = torch.tensor(x).float()
        grads = torch.tensor(grads).float()
        result = self._predict(x)

        curr_lr, _ = adjust_learning_rate(original_lr=self.original_learning_rate, **kwargs)
        optimizer = optim.SGD(self.parameters(), lr=curr_lr, momentum=0.9)
        result.backward(gradient=grads)
        optimizer.step()
        optimizer.zero_grad()
        msg = f"[DEBUG] finished HostDannModel.backward with learning rate:{curr_lr}"
        LOGGER.debug(msg)

    def predict(self, x):
        return self._predict(x).detach().numpy()

    def _predict(self, x):
        msg = f"[DEBUG] HostDannModel.predict"
        LOGGER.debug(msg)
        print(msg)
        x = torch.tensor(x).float()
        LOGGER.debug(f"data:{x}, {x.shape}")
        return self._calculate_regional_output(x)

    def local_predict(self, x):
        pred = self._local_predict(x)
        pred_prob = torch.sigmoid(pred.flatten())
        return pred_prob.numpy().reshape(-1, 1)

    def local_evaluate(self, x, y):
        pred = self._local_predict(x)
        pred_prob = torch.sigmoid(pred.flatten())
        pred_y = torch.round(pred_prob).long()
        auc = roc_auc_score(y, pred_prob.tolist())
        acc = accuracy_score(y, pred_y.tolist())

        y = torch.tensor(y).reshape(-1, 1).type_as(pred)
        class_loss = self.classifier_criterion(pred, y)
        return {"loss": class_loss.item(), "auc": auc, "acc": acc}

    def _local_predict(self, x):
        x = torch.tensor(x).float()
        output = self._calculate_regional_output(x)
        pred = self.classifier(output)
        return pred

    def _calculate_regional_output(self, x):
        wide_list, deep_par_list = self.partition_data_fn(x)
        output_list = []
        if len(deep_par_list) == 0:
            output_list = wide_list
        else:
            LOGGER.debug(f"regional_model_list len:{len(self.regional_model_list)}")
            LOGGER.debug(f"deep_par_list len:{len(deep_par_list)}")
            for regional_model, data_par in zip(self.regional_model_list, deep_par_list):
                embedding = self._combine_features(data_par)
                output_list.append(regional_model.compute_output(embedding))
            output_list = wide_list + output_list if len(wide_list) > 0 else output_list
        # LOGGER.debug(f"output_list:{output_list}")
        output = torch.cat(output_list, dim=1)
        LOGGER.debug(f"regional output combined:{output.shape}")
        return output

    def calculate_domain_discriminator_correctness(self, x, is_source=True):
        _, deep_par_list = self.partition_data_fn(x)
        output_list = []
        for wrapper, data_par in zip(self.regional_model_list, deep_par_list):
            embedding = self._combine_features(data_par)
            output_list.append(wrapper.calculate_domain_discriminator_correctness(embedding, is_source=is_source))
        # print(f"[DEBUG] is_source:{is_source}\t domain_discriminator_correctness {output_list}")
        return output_list


class RegionalModel(object):

    def __init__(self, extractor, aggregator, discriminator):
        self.extractor = extractor
        self.aggregator = aggregator
        self.discriminator = discriminator
        self.discriminator_set = False if discriminator is None else True
        self.discriminator_criterion = nn.CrossEntropyLoss()

    def state_dict(self):
        state_dict = dict()
        state_dict["extractor"] = self.extractor.state_dict()
        state_dict["aggregator"] = self.aggregator.state_dict()
        state_dict["discriminator"] = self.discriminator.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        self.extractor.load_state_dict(state_dict["extractor"])
        self.aggregator.load_state_dict(state_dict["aggregator"])
        discriminator_state_dict = state_dict["discriminator"]
        if self.discriminator is not None and discriminator_state_dict is not None:
            self.discriminator.load_state_dict(discriminator_state_dict)

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

    def print_parameters(self):
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
        alpha = kwargs.get("alpha")

        if alpha is None:
            raise Exception("alpha should not be None")

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

        LOGGER.debug(f"domain_feat shape:{domain_feat.shape}")
        LOGGER.debug(f"domain_labels shape:{domain_labels.shape}")
        LOGGER.debug(f"domain_output shape:{domain_output.shape}")

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
