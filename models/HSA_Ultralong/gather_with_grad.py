import torch
import torch.distributed as dist
from einops import rearrange
import traceback


class AllGatherWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group, dim=1):
        # assert len(tensor.shape) == 3, 'The input tensor is supposed to be (N, L, d)'
        # print(f'input tensor: {tensor.shape}')
        ctx.group = group
        ctx.dim = dim
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered, tensor, group=group)
        # return tuple(gathered)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_outputs):
        # grad_outputs: (N, P * L, d)
        # print(f'grad_outputs shape: {grad_outputs.shape}')
        ranks = dist.get_process_group_ranks(ctx.group)
        P = len(ranks)
        grad_outputs = rearrange(grad_outputs, 'N (P L) h d->P N L h d', P = len(ranks))
        grad_outputs = grad_outputs.contiguous()
        receive_tensor = torch.empty_like(grad_outputs)
        dist.all_to_all_single(receive_tensor, grad_outputs, group=ctx.group)
        sum_grad = receive_tensor.sum(dim=0)  # (N, L, ...)

        return sum_grad, None, None