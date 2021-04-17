# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 17:10:52

import torch
import ipdb


class Identity(torch.nn.Module):
    """Identity module
    
    x = x
    """

    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, consensus_type, dim=1):
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        ctx.shape = input_tensor.size()
        
        if ctx.consensus_type == 'avg':
            output = input_tensor.mean(dim=ctx.dim, keepdim=True)
        elif ctx.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        #ipdb.set_trace()
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in, None, None


class ConsensusModule(torch.nn.Module):
    """
    """

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, x):
        return SegmentConsensus.apply(x, self.consensus_type, self.dim)
