# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch implementation of composite gated delta update op."""

import torch

from ._utils import Version


class GatedDeltaUpdate(torch.nn.Module):
    """Gated delta update composite op."""

    def __init__(self, use_qk_l2_norm: bool = True) -> None:
        super().__init__()
        self.use_qk_l2_norm = use_qk_l2_norm
        self.version = Version.v1

    def forward(  # noqa: PLR0913
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform gated delta update."""

        def l2norm(x: torch.Tensor) -> torch.Tensor:
            return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)

        input_dtype = query.dtype

        # l2 norm
        if self.use_qk_l2_norm:
            query = l2norm(query)
            key = l2norm(key)

        # delta update rule needs to be computed in fp32
        query, key, value, beta, g = [
            x.to(torch.float32) for x in (query, key, value, beta, g)
        ]
        query = query * (query.shape[-1] ** -0.5)

        # initialize the output / state
        b, h, s, _dk = key.shape
        dv = value.shape[-1]
        output = torch.zeros(b, h, s, dv, dtype=torch.float32, device=query.device)
        state = initial_state.to(torch.float32)

        # delta update
        g_exp = g.exp()

        def cond_fn(
            t: torch.Tensor,
            state: torch.Tensor,
            output: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            ge: torch.Tensor,
            b_: torch.Tensor,
        ) -> torch.Tensor:
            return t < k.shape[2]

        def body_fn(
            t: torch.Tensor,
            state: torch.Tensor,
            output: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            ge: torch.Tensor,
            b_: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            t_idx = t.view(1)

            q_t = torch.index_select(q, 2, t_idx).transpose(2, 3)
            k_t = torch.index_select(k, 2, t_idx).transpose(2, 3)
            v_t = torch.index_select(v, 2, t_idx).squeeze(2)
            g_t = torch.index_select(ge, 2, t_idx).unsqueeze(-1)
            beta_t = torch.index_select(b_, 2, t_idx)

            state = state * g_t
            kv_mem = (state * k_t).sum(dim=-2)
            delta = ((v_t - kv_mem) * beta_t).unsqueeze(-2)
            state = state + k_t * delta

            output_value = (state * q_t).sum(dim=-2)
            indices = t.view(1, 1, 1, 1).expand(b, h, 1, dv)
            new_output = output.scatter(2, indices, output_value.unsqueeze(2))

            return t + 1, state, new_output

        # Run the while loop: equivalent to for t in range(s).
        # query/key/value/g_exp/beta are passed as additional_inputs so they are
        # explicit graph inputs to the subgraph rather than closed-over tensors.
        _, state, output = torch.ops.higher_order.while_loop(
            cond_fn,
            body_fn,
            (
                torch.tensor(0, device=query.device),
                state,
                output,
            ),
            (query, key, value, g_exp, beta),
        )

        return output.transpose(1, 2).to(input_dtype), state.to(input_dtype)
