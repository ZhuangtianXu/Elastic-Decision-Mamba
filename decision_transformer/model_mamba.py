import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from decision_transformer.utils import encode_return, get_d4rl_dataset_stats

from typing import Union
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba

@dataclass
class ModelArgs:
    d_model: int
    #n_layer: int
    #vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        #if self.vocab_size % self.pad_vocab_size_multiple != 0:
        #    self.vocab_size += (self.pad_vocab_size_multiple
        #                        - self.vocab_size % self.pad_vocab_size_multiple)



class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1].

        Note: the official repo chains residual blocks that look like
            [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
        where the first Add is a no-op. This is purely for performance reasons as this
        allows them to fuse the Add->Norm.

        We instead will realize our blocks as the more familiar, simpler, and numerically equivalent
            [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
        """
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)


        # ===== added from the original: see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L82
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = args.dt_rank**-0.5 * args.dt_scale
        if args.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif args.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(args.d_inner) * (math.log(args.dt_max) - math.log(args.dt_min)) + math.log(args.dt_min)).clamp(min=args.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        # ===== =====


        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)


    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output


    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        return y


    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        #! Note that the below is sequential, while the official implementation does a much faster
        #! parallel scan that is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):  #! get much slower for bigger l (= max_length K)
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
    


class Block(nn.Module):
    def __init__(self, hidden_size, drop_p, n_inner=None, layer_norm_epsilon=1e-5, model_type='dmamba'):  #, scale=False
        super().__init__()
        inner_dim = n_inner if n_inner is not None else 4 * hidden_size

        if model_type == 'dmamba':
            self.norm_mamba = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.mamba = Mamba(hidden_size)
        if model_type == 'dmamba-min':
            self.norm_mamba = RMSNorm(hidden_size)
            self.mamba = MambaBlock(ModelArgs(d_model=hidden_size))

        self.ln_2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.mlp_channels = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_size),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm_mamba(x))
        x = x + self.mlp_channels(self.ln_2(x))
        return x


class DecisionMamba(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_blocks,
        h_dim,
        context_len,
        n_heads,
        drop_p,
        env_name,
        max_timestep=4096,
        num_bin=120,
        dt_mask=False,
        rtg_scale=1000,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.num_bin = num_bin
        # for return scaling
        self.env_name = env_name
        self.rtg_scale = rtg_scale

        ### transformer blocks
        input_seq_len = 4 * context_len
        blocks = [
            Block(
                h_dim,
                drop_p,
                model_type='dmamba'
            )
            for _ in range(n_blocks)
        ]
        self.mamba = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.embed_reward = torch.nn.Linear(1, h_dim)

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True  # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, int(num_bin))
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(h_dim, act_dim)]
                + ([nn.Tanh()] if use_action_tanh else [])
            )
        )
        self.predict_reward = torch.nn.Linear(h_dim, 1)

    def forward(self, timesteps, states, actions, returns_to_go, rewards):

        B, T, _ = states.shape

        returns_to_go = returns_to_go.float()
        returns_to_go = (
            encode_return(
                self.env_name, returns_to_go, num_bin=self.num_bin, rtg_scale=self.rtg_scale
            )
            - self.num_bin / 2
        ) / (self.num_bin / 2)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        rewards_embeddings = self.embed_reward(rewards) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = (
            torch.stack(
                (
                    state_embeddings,
                    returns_embeddings,
                    action_embeddings,
                    rewards_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, 4 * T, self.h_dim)
        )

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.mamba(h)

        h = h.reshape(B, T, 4, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 0])  # predict next rtg given s
        state_preds = self.predict_state(
            h[:, 3]
        )  # predict next state given s, R, a, r
        action_preds = self.predict_action(
            h[:, 1]
        )  # predict action given s, R
        reward_preds = self.predict_reward(
            h[:, 2]
        )  # predict reward given s, R, a

        return state_preds, action_preds, return_preds, reward_preds


# a version that does not use reward at all
class ElasticDecisionMamba(
    DecisionMamba
):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_blocks,
        h_dim,
        context_len,
        n_heads,
        drop_p,
        env_name,
        max_timestep=4096,
        num_bin=120,
        dt_mask=False,
        rtg_scale=1000,
        num_inputs=3,
        real_rtg=False,
        is_continuous=True, # True for continuous action
    ):
        super().__init__(
            state_dim,
            act_dim,
            n_blocks,
            h_dim,
            context_len,
            n_heads,
            drop_p,
            env_name,
            max_timestep=max_timestep,
            num_bin=num_bin,
            dt_mask=dt_mask,
            rtg_scale=rtg_scale,
        )

        # return, state, action
        self.num_inputs = num_inputs
        self.is_continuous = is_continuous
        input_seq_len = num_inputs * context_len
        blocks = [
            Block(
                h_dim,
                drop_p,
                model_type='dmamba'
            )
            for _ in range(n_blocks)
        ]
        self.mamba = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        if not self.is_continuous:
            self.embed_action = torch.nn.Embedding(18, h_dim)
        else:
            self.embed_action = torch.nn.Linear(act_dim, h_dim)

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, int(num_bin))
        self.predict_rtg2 = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim + act_dim, state_dim)
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(h_dim, act_dim)]
                + ([nn.Tanh()] if is_continuous else [])
            )
        )
        self.predict_reward = torch.nn.Linear(h_dim, 1)

    def forward(
        self, timesteps, states, actions, returns_to_go, *args, **kwargs
    ):
        B, T, _ = states.shape
        returns_to_go = returns_to_go.float()
        returns_to_go = (
            encode_return(
                self.env_name, returns_to_go, num_bin=self.num_bin, rtg_scale=self.rtg_scale
            )
            - self.num_bin / 2
        ) / (self.num_bin / 2)
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = (
            torch.stack(
                (
                    state_embeddings,
                    returns_embeddings,
                    action_embeddings,
                    # rewards_embeddings,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(B, self.num_inputs * T, self.h_dim)
        )

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.mamba(h)
        h = h.reshape(B, T, self.num_inputs, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 0])  # predict next rtg given s
        return_preds2 = self.predict_rtg2(
            h[:, 0]
        )  # predict next rtg with implicit loss
        action_preds = self.predict_action(
            h[:, 1]
        )  # predict action given s, R
        state_preds = self.predict_state(torch.cat((h[:, 1], action_preds), 2))
        reward_preds = self.predict_reward(
            h[:, 2]
        )  # predict reward given s, R, a

        return (
            state_preds,
            action_preds,
            return_preds,
            return_preds2,
            reward_preds,
        )
