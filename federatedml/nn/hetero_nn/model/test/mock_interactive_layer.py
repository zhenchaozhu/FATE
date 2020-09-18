import numpy as np

from arch.api.utils import log_utils
from federatedml.nn.hetero_nn.backend.pytorch.interactive.dense_model import PlainHostDenseModel
from federatedml.nn.hetero_nn.model.test.mock_models import MockBottomDenseModel
from federatedml.nn.hetero_nn.model.test.mock_models import get_data
from federatedml.nn.hetero_nn.util import random_number_generator
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class MockInterActiveGuestDenseLayer(object):

    def __init__(self, params=None, layer_config=None, model_builder=None):
        self.nn_define = layer_config
        self.layer_config = layer_config

        self.host_input_shape = None
        self.guest_input_shape = None
        self.model = None
        self.rng_generator = random_number_generator.RandomNumberGenerator()
        self.model_builder = model_builder
        self.transfer_variable = None
        self.learning_rate = params.interactive_layer_lr
        self.encrypted_host_dense_output = None

        self.encrypted_host_input = None
        self.guest_input = None
        self.guest_output = None
        self.host_output = None

        self.dense_output_data = None

        self.guest_model = None
        self.host_model = None

        self.partitions = 0

        self.host_bottom_model = MockBottomDenseModel(input_dim=4, output_dim=3, optimizer_param=None)

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition):
        self.partitions = partition

    def __build_model(self, restore_stage=False):

        self.host_model = PlainHostDenseModel()
        self.host_model.build(input_shape=self.host_input_shape,
                              internal_model_builder=self.model_builder,
                              restore_stage=restore_stage)
        self.host_model.set_learning_rate(self.learning_rate)

        self.guest_model = None

    # Guest interactive layer
    def forward(self, guest_input, epoch=0, batch=0):
        print("[DEBUG] Guest interactive layer start forward propagation of epoch {} batch {}".format(epoch, batch))

        # mimic getting data from host. The data is the output of host's local model
        host_input, _ = get_data()
        host_bottom_output = self.host_bottom_model.forward(host_input)

        print(f"[DEBUG] *host_bottom_output: {host_bottom_output}")

        if self.guest_model is None:
            LOGGER.info("building interactive layers' training model")
            self.host_input_shape = host_bottom_output.shape[1]
            self.guest_input_shape = guest_input.shape[1] if guest_input is not None else 0
            self.__build_model()

        host_dense_output = self.forward_interactive(host_bottom_output, epoch, batch)

        self.dense_output_data = host_dense_output

        # self.guest_output = guest_output
        self.host_output = host_dense_output

        # Here actually perform the forward activation of the whole dense layer (logistic regression layer)
        activation_out = self.host_model.forward_activation(self.dense_output_data)
        print(f"[DEBUG] *activation_out:{activation_out}")

        return activation_out

    # Guest interactive layer
    def backward(self, output_gradient, epoch, batch):
        print("[DEBUG] Guest interactive layer start backward propagation of epoch {} batch {}".format(epoch, batch))
        activation_backward = self.host_model.backward_activation()[0]
        activation_gradient = output_gradient * activation_backward
        print("[DEBUG] *activation_gradient:", activation_gradient)
        # LOGGER.debug("interactive layer update guest weight of epoch {} batch {}".format(epoch, batch))
        # guest_input_gradient = self.update_guest(activation_gradient)
        #
        host_weight_gradient = self.backward_interactive(activation_gradient, epoch, batch)
        #
        host_input_gradient = self.update_host(activation_gradient, host_weight_gradient)
        print(f"[DEBUG] host_input_gradient:{host_input_gradient} to be sent to host")
        #
        # self.send_host_backward_to_host(host_input_gradient.get_obj(), epoch, batch)

        return None

    def send_host_backward_to_host(self, host_error, epoch, batch):
        self.transfer_variable.host_backward.remote(host_error,
                                                    role=consts.HOST,
                                                    idx=0,
                                                    suffix=(epoch, batch,))

    def update_guest(self, activation_gradient):
        # input_gradient = self.guest_model.get_input_gradient(activation_gradient)
        # weight_gradient = self.guest_model.get_weight_gradient(activation_gradient)
        # self.guest_model.apply_update(weight_gradient)

        # return gradient to be sent to guest bottom model
        return None

    def update_host(self, activation_gradient, weight_gradient):
        # activation_gradient_tensor = PaillierTensor(ori_data=activation_gradient, partitions=self.partitions)
        input_gradient = self.host_model.get_input_gradient(activation_gradient)
        # input_gradient = self.host_model.get_input_gradient(activation_gradient, acc_noise)
        print(f"[DEBUG] *input_gradient:{input_gradient}")

        self.host_model.update_weight(weight_gradient)
        self.host_model.update_bias(activation_gradient)

        # return gradient to be sent to host bottom model
        return input_gradient

    def forward_interactive(self, host_bottom_output, epoch, batch):
        print("[DEBUG] Guest get encrypted dense output of host model of epoch {} batch {}".format(epoch, batch))
        host_dense_output = self.host_model.forward_dense(host_bottom_output)
        print(f"[DEBUG] *host_dense_output: {host_dense_output}")
        return host_dense_output

    def backward_interactive(self, activation_gradient, epoch, batch):
        print("[DEBUG] Guest backward_interactive of epoch {} batch {}".format(epoch, batch))
        weight_gradient = self.host_model.get_weight_gradient(activation_gradient)
        print(f"[DEBUG] *lr_weight_gradient:{weight_gradient}")

        # noise_w = self.rng_generator.generate_random_number(encrypted_weight_gradient.shape)
        # self.transfer_variable.encrypted_guest_weight_gradient.remote(encrypted_weight_gradient + noise_w,
        #                                                               role=consts.HOST,
        #                                                               idx=-1,
        #                                                               suffix=(epoch, batch,))
        #
        # LOGGER.info("get decrypted weight graident of epoch {} batch {}".format(epoch, batch))
        # decrypted_weight_gradient = self.transfer_variable.decrypted_guest_weight_gradient.get(idx=0,
        #                                                                                        suffix=(epoch, batch,))
        # decrypted_weight_gradient -= noise_w
        # encrypted_acc_noise = self.get_encrypted_acc_noise_from_host(epoch, batch)

        return weight_gradient

    # def get_host_encrypted_forward_from_host(self, epoch, batch):
    #     return self.transfer_variable.encrypted_host_forward.get(idx=0,
    #                                                              suffix=(epoch, batch,))
    #
    # def send_guest_encrypted_forward_output_with_noise_to_host(self, encrypted_guest_forward_with_noise, epoch, batch):
    #     return self.transfer_variable.encrypted_guest_forward.remote(encrypted_guest_forward_with_noise,
    #                                                                  role=consts.HOST,
    #                                                                  idx=-1,
    #                                                                  suffix=(epoch, batch,))
    #
    # def get_guest_decrypted_forward_from_host(self, epoch, batch):
    #     return self.transfer_variable.decrypted_guest_fowrad.get(idx=0,
    #                                                              suffix=(epoch, batch,))
    #
    # def get_encrypted_acc_noise_from_host(self, epoch, batch):
    #     return self.transfer_variable.encrypted_acc_noise.get(idx=0,
    #                                                           suffix=(epoch, batch,))
    #
    # def get_output_shape(self):
    #     return self.host_model.output_shape

    def export_model(self):
        return None

    def restore_model(self, interactive_layer_param):
        pass


class MockInteractiveHostDenseLayer(object):
    def __init__(self, param):
        # self.acc_noise = None
        # self.learning_rate = param.interactive_layer_lr
        # self.encrypted_mode_calculator_param = param.encrypted_model_calculator_param
        # self.encrypter = self.generate_encrypter(param)
        # self.train_encrypted_calculator = []
        # self.predict_encrypted_calculator = []
        self.transfer_variable = None
        self.partitions = 1
        # self.input_shape = None
        # self.output_unit = None
        # self.rng_generator = random_number_generator.RandomNumberGenerator()

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    # def generated_encrypted_calculator(self):
    #     encrypted_calculator = EncryptModeCalculator(self.encrypter,
    #                                                  self.encrypted_mode_calculator_param.mode,
    #                                                  self.encrypted_mode_calculator_param.re_encrypted_rate)
    #
    #     return encrypted_calculator

    def set_partition(self, partition):
        self.partitions = partition

    # Host interactive layer
    def forward(self, host_input, epoch=0, batch=0):
        print("[DEBUG] MockInteractiveHostDenseLayer.forward")
        pass

    def backward(self, epoch, batch):
        """
           return the gradient to be back-propagated to host bottom model. The gradient is in the format of numpy
        """
        print("[DEBUG] MockInteractiveHostDenseLayer.backward")
        host_input_gradient = np.array([[-0.02694966, -0.02226276, -0.16404139],
                                        [0.08331229, 0.06882319, 0.50711824]])
        return host_input_gradient

    # def send_encrypted_acc_noise_to_guest(self, encrypted_acc_noise, epoch, batch):
    #     self.transfer_variable.encrypted_acc_noise.remote(encrypted_acc_noise,
    #                                                       idx=0,
    #                                                       role=consts.GUEST,
    #                                                       suffix=(epoch, batch,))
    #
    # def get_guest_encrypted_weight_gradient_from_guest(self, epoch, batch):
    #     encrypted_guest_weight_gradient = self.transfer_variable.encrypted_guest_weight_gradient.get(idx=0,
    #                                                                                                  suffix=(
    #                                                                                                      epoch, batch,))
    #
    #     return encrypted_guest_weight_gradient
    #
    # def send_host_encrypted_forward_to_guest(self, encrypted_host_input, epoch, batch):
    #     self.transfer_variable.encrypted_host_forward.remote(encrypted_host_input,
    #                                                          idx=0,
    #                                                          role=consts.GUEST,
    #                                                          suffix=(epoch, batch,))
    #
    # def send_guest_decrypted_weight_gradient_to_guest(self, decrypted_guest_weight_gradient, epoch, batch):
    #     self.transfer_variable.decrypted_guest_weight_gradient.remote(decrypted_guest_weight_gradient,
    #                                                                   idx=0,
    #                                                                   role=consts.GUEST,
    #                                                                   suffix=(epoch, batch,))
    #
    # def get_host_backward_from_guest(self, epoch, batch):
    #     host_backward = self.transfer_variable.host_backward.get(idx=0,
    #                                                              suffix=(epoch, batch,))
    #
    #     return host_backward
    #
    # def get_guest_encrypted_forwrad_from_guest(self, epoch, batch):
    #     encrypted_guest_forward = self.transfer_variable.encrypted_guest_forward.get(idx=0,
    #                                                                                  suffix=(epoch, batch,))
    #
    #     return encrypted_guest_forward
    #
    # def send_decrypted_guest_forward_with_noise_to_guest(self, decrypted_guest_forward_with_noise, epoch, batch):
    #     self.transfer_variable.decrypted_guest_fowrad.remote(decrypted_guest_forward_with_noise,
    #                                                          idx=0,
    #                                                          role=consts.GUEST,
    #                                                          suffix=(epoch, batch,))
    #
    # def generate_encrypter(self, param):
    #     LOGGER.info("generate encrypter")
    #     if param.encrypt_param.method.lower() == consts.PAILLIER.lower():
    #         encrypter = PaillierEncrypt()
    #         encrypter.generate_key(param.encrypt_param.key_length)
    #     else:
    #         raise NotImplementedError("encrypt method not supported yet!!!")
    #
    #     return encrypter

    def export_model(self):
        # interactive_layer_param = InteractiveLayerParam()
        # interactive_layer_param.acc_noise = pickle.dumps(self.acc_noise)
        return None

    def restore_model(self, interactive_layer_param):
        pass
