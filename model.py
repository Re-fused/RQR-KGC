"""RQR-KGC model used in the paper."""

import torch
from torch import nn


def hamilton_product(left, right):
    """Block-wise Hamilton product in component-major layout."""
    a, b, c, d = torch.chunk(left, 4, dim=-1)
    e, f, g, h = torch.chunk(right, 4, dim=-1)
    return torch.cat((
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g + c * e + d * f - b * h,
        a * h + d * e + b * g - c * f,
    ), dim=-1)


def normalize_quaternion(value):
    components = torch.stack(torch.chunk(value, 4, dim=-1), dim=-1)
    components = components / components.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.cat(tuple(components.unbind(dim=-1)), dim=-1)


def quaternion_conjugate(value):
    real, i, j, k = torch.chunk(value, 4, dim=-1)
    return torch.cat((real, -i, -j, -k), dim=-1)


class RQRMKGC(nn.Module):
    """Relative quaternion transformations with reciprocal relation modeling."""

    def __init__(self, nentity, nrelation, hidden_dim, gamma=10.0,
                 residual_bound=0.1, endpoint_bound=0.5):
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four")
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.residual_bound = float(residual_bound)
        self.endpoint_bound = float(endpoint_bound)
        self.register_buffer("gamma", torch.tensor(float(gamma)))
        embedding_range = (float(gamma) + 2.0) / hidden_dim
        self.entity_embedding = nn.Parameter(torch.empty(nentity, hidden_dim))
        self.relation_embedding = nn.Parameter(torch.empty(nrelation, 3 * hidden_dim))
        self.forward_residual = nn.Parameter(torch.zeros(nrelation, 3 * hidden_dim))
        self.reverse_residual = nn.Parameter(torch.zeros(nrelation, 3 * hidden_dim))
        self.endpoint_scale_raw = nn.Parameter(
            torch.zeros(nrelation, 2, hidden_dim // 4))
        nn.init.uniform_(self.entity_embedding, -embedding_range, embedding_range)
        nn.init.uniform_(self.relation_embedding, -embedding_range, embedding_range)

    @staticmethod
    def decode(relation):
        source_raw, translation, relative_raw = torch.chunk(relation, 3, dim=-1)
        source = normalize_quaternion(source_raw)
        relative = normalize_quaternion(relative_raw)
        target = normalize_quaternion(hamilton_product(source, relative))
        return source, translation, relative, target

    def analytic_inverse(self, relation):
        _, translation, relative, target = self.decode(relation)
        return torch.cat((target, -translation, quaternion_conjugate(relative)), dim=-1)

    def effective_relation(self, relation_ids, reverse=False):
        relation = self.relation_embedding[relation_ids]
        if reverse:
            relation = self.analytic_inverse(relation)
            residual = self.reverse_residual[relation_ids]
        else:
            residual = self.forward_residual[relation_ids]
        return relation + self.residual_bound * torch.tanh(residual)

    def endpoint_scales(self, relation_ids, reverse=False):
        scales = torch.exp(
            self.endpoint_bound * torch.tanh(self.endpoint_scale_raw[relation_ids]))
        if reverse:
            scales = scales.flip(1)
        return scales.repeat(1, 1, 4)

    def forward(self, sample, mode="single"):
        if mode in ("single", "head-single", "tail-single"):
            positive = sample
            head = self.entity_embedding[positive[:, 0]].unsqueeze(1)
            tail = self.entity_embedding[positive[:, 2]].unsqueeze(1)
        elif mode in ("head-batch", "tail-batch"):
            positive, candidates = sample
            head = (self.entity_embedding[candidates] if mode == "head-batch"
                    else self.entity_embedding[positive[:, 0]].unsqueeze(1))
            tail = (self.entity_embedding[candidates] if mode == "tail-batch"
                    else self.entity_embedding[positive[:, 2]].unsqueeze(1))
        else:
            raise ValueError(f"unsupported mode: {mode}")

        relation_ids = positive[:, 1]
        reverse = mode in ("head-single", "head-batch")
        relation = self.effective_relation(relation_ids, reverse).unsqueeze(1)
        if reverse:
            head, tail = tail, head
        source, translation, _, target = self.decode(relation)
        scales = self.endpoint_scales(relation_ids, reverse)
        distance = (
            scales[:, :1] * hamilton_product(head, source)
            + translation
            - scales[:, 1:] * hamilton_product(tail, target)
        ).norm(p=1, dim=-1)
        return self.gamma - distance

