from __future__ import absolute_import, division, print_function
import torch.nn as nn

def prunable(key, arch):
    if 'resnet' in arch:
        if 'fc' in key:
            return False
        elif not ('layer' in key):
            return False
        elif 'downsample' in key:
            return False
        return True
    return False


def gen_percentages(model, arch, prune_ratio=0.):
    # This function is used to generate the target sparsity for
    # specific network. Here I am using AlexNet as an example.
    # It needs to be extended to support more networks. And a
    # smart method to determine the sparsity of each layer should
    # be defined.
    if arch == 'alexnet':
        # mannually define the sparsities
        percentages_dict = {
                'features.module.0.weight'  : 0.00,
                'features.module.3.weight'  : 0.80,
                'features.module.6.weight'  : 0.80,
                'features.module.8.weight'  : 0.80,
                'features.module.10.weight' : 0.80,
                'classifier.1.weight'       : 0.95,
                'classifier.4.weight'       : 0.9,
                'classifier.6.weight'       : 0.85,
                }
        return percentages_dict
    elif arch == 'resnet18':
        # determine the actual prune_ratio for target layers
        params = model.state_dict()
        prunable_weights = 0
        unprunable_weights = 0
        for key in params:
            if ('weight' in key) and (len(params[key].shape) > 1):
                if prunable(key, arch):
                    prunable_weights += int(params[key].nelement())
                else:
                    unprunable_weights += int(params[key].nelement())
        prune_ratio_layer = float(prunable_weights + unprunable_weights) * prune_ratio / \
                float(prunable_weights)
        percentages_dict = {}
        for key in params:
            if ('weight' in key) and (len(params[key].shape) > 1):
                if prunable(key, arch):
                    percentages_dict[key] = prune_ratio_layer
                else:
                    percentages_dict[key] = 0.
        return percentages_dict
    else:
        raise Exception ('{} not supported'.format(arch))
    return None

class admm_op():
    #
    # The operator for ADMM stage of the pruning process.
    #
    def __init__(self, model, percentages, admm_iter, pho):
        #
        # Params:
        #     percentages: list of target sparsity for each layer
        #     admm_iter: admm is performed every #admm_iter epochs
        #     pho: pho value
        #
        self.model = model
        self.W = []
        self.U = []
        self.Z = []
        self.percentages = []
        for key, value in model.named_parameters():
            if key in percentages:
                self.W.append(value)
                self.Z.append(value.data.clone())
                self.U.append(value.data.clone().zero_())
                self.percentages.append(percentages[key])
        self.admm_iter = admm_iter
        self.pho = pho
        return

    def update(self, epoch):
        #
        # This func is for updating W, U and Z tensors.
        #
        if epoch % self.admm_iter == 0:
            print('\n!!! Updating U,Z ...')
            for index in range(len(self.W)):
                # calculate threshold
                importance = self.W[index].data.add(self.U[index]).abs()
                n = int(self.percentages[index] * self.W[index].nelement())
                if n > 0:
                    threshold = float(importance.flatten().cpu().kthvalue(n - 1)[0].cpu())
                else:
                    threshold = -1.0
                mask = importance.gt(threshold).float()

                # update Z
                self.Z[index] = self.W[index].data.add(self.U[index]).mul(mask)

                # update U
                if epoch > 0:
                    # For epoch = 0, U is fixed to all zero.
                    self.U[index] = self.U[index] + self.W[index].data - self.Z[index]

                target_mask_sum = \
                        self.U[index].sub(self.Z[index]).mul(1.0 - mask).abs().sum()
                print('[{:3d}] target_mask_sum: {:9.3f}'.format(index,
                    target_mask_sum))
        print('\n')
        return

    def print_info(self):
        #
        # Print the necessary information through the ADMM process:
        #     target_sparsity: the target sparsity for each layer/network
        #     small_sparsity: to monitor how sparse the current weight is, I define
        #                     the small_sparsity to be percentage of |weight| < 1e-3
        #
        print('\n' + '-' * 30)
        total_W_Z_error = 0.
        total_small = 0
        total_pruned = 0
        total_weight = 0
        print('pho:', self.pho)
        for index in range(len(self.W)):
            W_Z_error = self.W[index].data.sub(self.Z[index]).pow(2.0).sum()
            target_sparsity = self.Z[index].eq(0.).float().sum().div(
                    self.Z[index].nelement())
            small_sparsity = self.W[index].data.abs().lt(1e-3).float().sum()
            total_small += int(small_sparsity)
            small_sparsity = small_sparsity / self.Z[index].nelement()
            total_pruned += int(self.Z[index].eq(0.).float().sum())
            total_weight += int(self.Z[index].nelement())
            total_W_Z_error += W_Z_error
            print('[{:3d}] shape: {:30}\t W_Z_error: {:9.3f} target_sparsity: {:7.3f} small_sparsity: {:7.3f}'
                    .format(index,
                        self.W[index].data.shape,
                        W_Z_error,
                        target_sparsity,
                        small_sparsity))
        print('\n[Total] W_Z_error: {:6.3f} target_sparsity: {:7.3f} small_sparsity: {:7.3f}\n'
                .format(total_W_Z_error,
                    float(total_pruned) / float(total_weight),
                    float(total_small) / float(total_weight)))
        print('-' * 30 + '\n')
        return

    def loss_grad(self):
        #
        # The regularization of term pho/2*|W-Z+U|^2 is applied in a similar way
        # of how PyTorch implements L2-Norm weight decay.
        #
        for index in range(len(self.W)):
            grad = self.W[index].data.sub(self.Z[index]).add(self.U[index])
            grad = grad.mul(self.pho)
            self.W[index].grad.data += grad

class retrain_op():
    #
    # The operator for RETRAIN stage of the pruning process.
    #
    def __init__(self, model, percentages):
        self.model = model
        self.percentages = []
        self.W = []
        self.mask = []
        # generate the mask list
        for key, value in model.named_parameters():
            if key in percentages:
                self.W.append(value)
                importance = value.data.abs()
                n = int(percentages[key] * value.nelement())
                if n > 0:
                    threshold = float(importance.flatten().cpu().kthvalue(n - 1)[0].cpu())
                else:
                    threshold = -1.0
                mask_tmp = importance.gt(threshold).float()
                self.mask.append(mask_tmp)
        return

    def apply_mask(self):
        # apply the mask for each layer
        for index in range(len(self.W)):
            self.W[index].data = self.W[index].data.mul(self.mask[index])
        return

    def print_info(self):
        # print the sparsity info for each layer/network
        print('\n' + '-' * 30)
        pruned_total = 0
        weight_total = 0
        for index in range(len(self.W)):
            pruned_tmp = int(self.W[index].data.eq(0).sum())
            weight_tmp = int(self.W[index].data.nelement())
            pruned_total += pruned_tmp
            weight_total += weight_tmp
            print('[{:3d}] {:11d}/{:11d} = {:7.3f}%'.format(
                index,
                pruned_tmp, weight_tmp,
                float(pruned_tmp) / float(weight_tmp)
                ))
        print('\nTotal {:11d}/{:11d} = {:7.3f}%\n'.format(
            pruned_total, weight_total,
            float(pruned_total) / float(weight_total)
            ))
        print('-' * 30 + '\n')
        return
