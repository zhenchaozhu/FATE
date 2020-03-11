from federatedml.param.cwjtestparam import CwjParam
from arch.api.utils import log_utils
import numpy as np
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.feature.sparse_vector import SparseVector
from federatedml.model_base import ModelBase
from federatedml.feature.fate_element_type import NoneType



LOGGER = log_utils.getLogger()

class CWJBase(ModelBase):

    def __init__(self):
        super(CWJBase, self).__init__()
        self.model_param = CwjParam()
        self.binning_obj = None
        LOGGER.debug('model')


    def _init_model(self,param):
        LOGGER.info('initialization done')

    @staticmethod
    def data_format_transform(row):
        if type(row.features).__name__ != consts.SPARSE_VECTOR:
            feature_shape = row.features.shape[0]
            indices = []
            data = []

            for i in range(feature_shape):
                if np.isnan(row.features[i]):
                    indices.append(i)
                    data.append(NoneType())
                elif np.abs(row.features[i]) < consts.FLOAT_ZERO:
                    continue
                else:
                    indices.append(i)
                    data.append(row.features[i])

            row.features = SparseVector(indices, data, feature_shape)
        else:
            sparse_vec = row.features.get_sparse_vector()
            for key in sparse_vec:
                if sparse_vec.get(key) == NoneType() or np.isnan(sparse_vec.get(key)):
                    sparse_vec[key] = NoneType()

            row.features.set_sparse_vector(sparse_vec)

        return row

    def data_alignment(self, data_inst):
        abnormal_detection.empty_table_detection(data_inst)
        abnormal_detection.empty_feature_detection(data_inst)

        schema = data_inst.schema
        new_data_inst = data_inst.mapValues(lambda row: CWJBase.data_format_transform(row))

        new_data_inst.schema = schema

        return new_data_inst

    def fit(self, train=None, validate=None):
        LOGGER.debug('fit called')
        data_inst = self.data_alignment(train)
        if train is not None:
            LOGGER.debug('showing inst_num:{}'.format(data_inst.count()))
        binning_result = self.binning_obj.average_run(data_instances=data_inst, bin_num=20)
        LOGGER.debug('binning result is {}'.format(binning_result))


