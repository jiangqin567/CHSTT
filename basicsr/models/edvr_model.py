import logging
import torch
from torch.nn.parallel import DistributedDataParallel

from basicsr.models.video_base_model import VideoBaseModel
from basicsr.models.sr_model import SRModel
import pdb

logger = logging.getLogger('basicsr')


class EDVRModel(VideoBaseModel):
# class EDVRModel(SRModel):
    """EDVR Model.

    Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.  # noqa: E501
    """

    def __init__(self, opt):
        super(EDVRModel, self).__init__(opt)
        if self.is_train:
            self.train_tsa_iter = opt['train'].get('tsa_iter')
            self.train_spynet_iter = opt['train'].get('spynet_iter')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        logger.info(f'Multiple the learning rate for dcn with {dcn_lr_mul}.')

        if dcn_lr_mul == 1:
            # optim_params = self.net_g.parameters()
            optim_params = []
            for name, param in self.net_g.named_parameters():

                # param.requires_grad = False if ('spynet' in name) or ('feat_extract' in name) or ('transformer' in name) or ('reconstruction' in name) else True
                # param.requires_grad = False if ('spynet' in name) or ('transformer' in name) else True
                # param.requires_grad = False if 'transformer' in name else True
                # param.requires_grad = False if 'spynet' in name else True

                if param.requires_grad:
                    optim_params.append(param)

                else:
                    logger.info(f'Params {name} will not be optimized.')

        else:  # separate dcn params and normal params for differnet lr
            normal_params = []
            dcn_params = []
            for name, param in self.net_g.named_parameters():
                if 'dcn' in name:
                    dcn_params.append(param)

                else:
                    normal_params.append(param)

            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },

                {
                    'params': dcn_params,
                    'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.train_spynet_iter:
            if current_iter == 1:
                logger.info(f'Only train modules other than Spynet for {self.train_spynet_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name:
                        param.requires_grad = False
            elif current_iter == self.train_spynet_iter:
                logger.warning('Train all the parameters.')
                for param in self.net_g.parameters():
                    param.requires_grad = True
                if isinstance(self.net_g, DistributedDataParallel):
                    logger.warning('Set net_g.find_unused_parameters = False.')
                    self.net_g.find_unused_parameters = False

        if self.train_tsa_iter:
            if current_iter == 1:
                logger.info(f'Only train TSA module for {self.train_tsa_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'fusion' not in name:
                        param.requires_grad = False
            elif current_iter == self.train_tsa_iter:
                logger.warning('Train all the parameters.')
                for param in self.net_g.parameters():
                    param.requires_grad = True
                if isinstance(self.net_g, DistributedDataParallel):
                    logger.warning('Set net_g.find_unused_parameters = False.')
                    self.net_g.find_unused_parameters = False

        super(VideoBaseModel, self).optimize_parameters(current_iter)
