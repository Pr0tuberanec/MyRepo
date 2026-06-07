from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import expand_as_right, unravel_key_list
from torchrl.data.tensor_specs import Composite, Unbounded
from torchrl.modules import LSTMCell, MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from benchmarl.utils import DEVICE_TYPING

# Ключ глобального состояния в TensorDict (CAMAR: concat(local_obs))
GLOBAL_STATE_KEY = "state"


class HyperLSTMCell(nn.Module):
    """Weight-scaling HyperLSTM cell (Ha et al., 2016).

    gate = LN( d_h(z_h) ⊙ (W_h h) + d_x(z_x) ⊙ (W_x x) + d_b(z_b) )
    где z = Linear(h_hat), d = Linear(z), h_hat — выход малого LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hyper_size: int,
        n_z: int,
        bias: bool = True,
        hyper_global_state_size: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size

        # --- малый LSTM (hyper-network) ---
        # По умолчанию вход [h; x]. При hyper_use_global_state — [h; global_state].
        hyper_cell_input_size = (
            hyper_global_state_size if hyper_global_state_size > 0 else input_size
        )
        self.hyper_cell = LSTMCell(
            hidden_size + hyper_cell_input_size, hyper_size, bias=bias
        )

        self.n_z = n_z

        # --- проекции h_hat → z для 4 gate одной линейкой ---
        self.z_h = nn.Linear(hyper_size, 4 * n_z, bias=True)
        self.z_x = nn.Linear(hyper_size, 4 * n_z, bias=True)
        self.z_b = nn.Linear(hyper_size, 4 * n_z, bias=False)

        # --- генераторы масштабирования z → d (nn.Linear — auto-init) ---
        self.d_h = nn.ModuleList([nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)])
        self.d_x = nn.ModuleList([nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)])
        self.d_b = nn.ModuleList([nn.Linear(n_z, hidden_size, bias=True)  for _ in range(4)])

        # --- основные матрицы: один Parameter (4*H, H) вместо ParameterList ---
        # Это позволяет один F.linear вместо 4 einsum без torch.cat на каждый шаг
        k = hidden_size ** -0.5
        self.w_h = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.w_x = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        nn.init.uniform_(self.w_h, -k, k)
        nn.init.uniform_(self.w_x, -k, k)

        # --- LayerNorm на каждый gate и c_next ---
        self.layer_norm   = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.layer_norm_c = nn.LayerNorm(hidden_size)

    def forward(self, x, h, c, h_hat, c_hat, global_state=None):
        # x:(B,in), h/c:(B,hidden), h_hat/c_hat:(B,hyper)
        # global_state: если задан, подаётся в hyper-ветку вместо x
        B, H = h.shape

        # Шаг 1: hyper-LSTM — вход [h; global_state] или [h; x]
        hyper_cell_input = global_state if global_state is not None else x
        x_hat = torch.cat([h, hyper_cell_input], dim=-1)
        h_hat, c_hat = self.hyper_cell(x_hat, (h_hat, c_hat))

        # Шаг 2: z-проекции → (B, 4, n_z) через view
        z_h = self.z_h(h_hat).view(B, 4, self.n_z)
        z_x = self.z_x(h_hat).view(B, 4, self.n_z)
        z_b = self.z_b(h_hat).view(B, 4, self.n_z)

        # Шаг 3: d-проекции — стакуем веса nn.Linear для batched matmul
        # d_h[i].weight: (H, n_z); stack → (4, H, n_z); einsum "bgi,gHi->bgH"
        d_h = torch.einsum("bgi,gHi->bgH", z_h, torch.stack([l.weight for l in self.d_h]))
        d_x = torch.einsum("bgi,gHi->bgH", z_x, torch.stack([l.weight for l in self.d_x]))
        d_b = (torch.einsum("bgi,gHi->bgH", z_b, torch.stack([l.weight for l in self.d_b]))
               + torch.stack([l.bias for l in self.d_b]))  # (B, 4, H) + (4, H)

        # Шаг 4: основные matmul — один F.linear (w_h уже (4*H, H))
        wh_out = F.linear(h, self.w_h).view(B, 4, H)   # (B, 4*H) → (B, 4, H)
        wx_out = F.linear(x, self.w_x).view(B, 4, H)

        # Шаг 5: scaled pre-activations — все gate сразу
        y = d_h * wh_out + d_x * wx_out + d_b   # (B, 4, H)

        # Шаг 6: LayerNorm — стакуем параметры для векторизованного LN
        ln_w = torch.stack([ln.weight for ln in self.layer_norm])   # (4, H)
        ln_b = torch.stack([ln.bias   for ln in self.layer_norm])
        mean = y.mean(-1, keepdim=True)
        var  = y.var(-1, unbiased=False, keepdim=True)
        y    = (y - mean) * (var + 1e-5).rsqrt() * ln_w + ln_b     # (B, 4, H)
        # LSTM-уравнения
        i_g, f_g, g_g, o_g = y.unbind(dim=1)   # each (B, H)
        c_next = f_g.sigmoid() * c + i_g.sigmoid() * g_g.tanh()
        h_next = o_g.sigmoid() * self.layer_norm_c(c_next).tanh()

        return h_next, c_next, h_hat, c_hat


class HyperLSTM(torch.nn.Module):
    """Многослойный HyperLSTM: разворачивает ячейку по времени.

    Аналог класса LSTM в lstm.py — принимает (input, is_init, h, c, h_hat, c_hat),
    возвращает (output, h_n, c_n, h_hat_n, c_hat_n).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hyper_size: int,
        n_z: int,
        n_layers: int,
        bias: bool,
        device: DEVICE_TYPING,
        time_dim: int = -2,
        hyper_global_state_size: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size
        self.n_layers = n_layers
        self.time_dim = time_dim

        # Стек ячеек — аналог self.lstms в lstm.py:45
        # Первый слой принимает input_size, остальные — hidden_size
        self.cells = nn.ModuleList(
            [
                HyperLSTMCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    hyper_size,
                    n_z,
                    bias,
                    hyper_global_state_size=hyper_global_state_size,
                )
                for i in range(n_layers)
            ]
        ).to(device)

    def forward(self, input, is_init, h, c, h_hat, c_hat, global_state=None):
        # h, c:         (B, n_layers, hidden_size)
        # h_hat, c_hat: (B, n_layers, hyper_size)
        # is_init:      (B, seq, n_agents, 1) — уже expand-нут снаружи

        # Разбиваем по слоям — аналог lstm.py:60-61
        h     = list(h.unbind(dim=-2))
        c     = list(c.unbind(dim=-2))
        h_hat = list(h_hat.unbind(dim=-2))
        c_hat = list(c_hat.unbind(dim=-2))

        hs = []
        global_state_steps = (
            global_state.unbind(self.time_dim)
            if global_state is not None
            else [None] * input.shape[self.time_dim]
        )
        # Цикл по шагам времени — аналог lstm.py:63-77
        for in_t, init_t, global_state_t in zip(
            input.unbind(self.time_dim),
            is_init.unbind(self.time_dim),
            global_state_steps,
        ):
            for layer in range(self.n_layers):
                # Сброс hidden при начале нового эпизода — из lstm.py:67-68
                # Расширено на h_hat/c_hat: они тоже должны сбрасываться
                h[layer]     = torch.where(init_t, 0, h[layer])
                c[layer]     = torch.where(init_t, 0, c[layer])
                h_hat[layer] = torch.where(init_t, 0, h_hat[layer])
                c_hat[layer] = torch.where(init_t, 0, c_hat[layer])

                # Шаг ячейки — аналог lstm.py:70, но возвращает 4 состояния
                h[layer], c[layer], h_hat[layer], c_hat[layer] = self.cells[layer](
                    in_t,
                    h[layer],
                    c[layer],
                    h_hat[layer],
                    c_hat[layer],
                    global_state=global_state_t,
                )
                in_t = h[layer]

            hs.append(in_t)

        # Собираем обратно по слоям — аналог lstm.py:78-80
        return (
            torch.stack(hs, dim=self.time_dim),   # output
            torch.stack(h,     dim=-2),            # h_n
            torch.stack(c,     dim=-2),            # c_n
            torch.stack(h_hat, dim=-2),            # h_hat_n
            torch.stack(c_hat, dim=-2),            # c_hat_n
        )


def _get_hyper_net(
    input_size,
    hidden_size,
    hyper_size,
    n_z,
    n_layers,
    bias,
    device,
    compile,
    hyper_global_state_size=0,
):
    """Фабричная функция — аналог get_net() в lstm.py:85."""
    net = HyperLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        hyper_size=hyper_size,
        n_z=n_z,
        n_layers=n_layers,
        bias=bias,
        device=device,
        hyper_global_state_size=hyper_global_state_size,
    )
    if compile:
        net = torch.compile(net, mode="reduce-overhead")
    return net


class MultiAgentHyperLSTM(torch.nn.Module):
    """Мультиагентная обёртка над HyperLSTM.

    Полный аналог MultiAgentLSTM (lstm.py:99), расширенный на h_hat/c_hat.
    Поддерживает share_params (vmap по весам) и centralised (concat obs агентов).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hyper_size: int,
        n_z: int,
        n_agents: int,
        device: DEVICE_TYPING,
        centralised: bool,
        share_params: bool,
        n_layers: int,
        bias: bool,
        compile: bool,
        hyper_use_global_state: bool = False,
        global_state_size: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size
        self.n_agents = n_agents
        self.device = device
        self.centralised = centralised
        self.share_params = share_params
        self.n_layers = n_layers
        self.bias = bias
        self.compile = compile
        self.hyper_use_global_state = hyper_use_global_state
        # hyper_global_state_size: размерность global_state для hyper-ветки (0 → fallback на x)
        hyper_global_state_size = global_state_size if hyper_use_global_state else 0

        # Centralised critic: concat obs всех агентов — из lstm.py:125-126
        if self.centralised:
            input_size = input_size * self.n_agents

        # Создаём сети: одну (share_params) или по одной на агента — из lstm.py:128-139
        agent_networks = [
            _get_hyper_net(
                input_size,
                hidden_size,
                hyper_size,
                n_z,
                n_layers,
                bias,
                device,
                compile,
                hyper_global_state_size=hyper_global_state_size,
            )
            for _ in range(self.n_agents if not self.share_params else 1)
        ]
        self._make_params(agent_networks)

        # _empty_net на meta-device для vmap — из lstm.py:142-155
        # meta-device не выделяет память, нужен только граф вычислений для vmap
        with torch.device("meta"):
            self._empty_net = _get_hyper_net(
                input_size,
                hidden_size,
                hyper_size,
                n_z,
                n_layers,
                bias,
                "meta",
                compile,
                hyper_global_state_size=hyper_global_state_size,
            )
            TensorDict.from_module(self._empty_net).data.to("meta").to_module(self._empty_net)

    def _make_params(self, agent_networks):
        # Из lstm.py:275-279: упаковываем веса сетей в TensorDict-параметр
        # share_params=True → один набор весов; False → веса каждого агента отдельно
        if self.share_params:
            self.params = TensorDict.from_module(agent_networks[0], as_module=True)
        else:
            self.params = TensorDict.from_modules(*agent_networks, as_module=True)

    def forward(self, input, is_init, h_0=None, c_0=None, h_hat_0=None, c_hat_0=None, global_state=None):
        # Структура полностью из lstm.py:157-239, расширена на h_hat/c_hat

        assert is_init is not None
        # training=True когда hidden не передан (обучение на батче траекторий)
        # training=False когда hidden передан (сбор данных, один шаг за раз)
        training = h_0 is None

        # При инференсе без batch-размерности — добавляем её — из lstm.py:171-179
        missing_batch = False
        if not training and len(input.shape) < 3:
            missing_batch = True
            input   = input.unsqueeze(0)
            h_0     = h_0.unsqueeze(0)
            c_0     = c_0.unsqueeze(0)
            h_hat_0 = h_hat_0.unsqueeze(0)
            c_hat_0 = c_hat_0.unsqueeze(0)
            is_init = is_init.unsqueeze(0)
            if global_state is not None:
                global_state = global_state.unsqueeze(0)

        # При сборе данных эмулируем seq-dimension — из lstm.py:181-184
        if not training:
            input = input.unsqueeze(1)
            if global_state is not None:
                global_state = global_state.unsqueeze(1)

        batch = input.shape[0]
        seq   = input.shape[1]
        assert input.shape == (batch, seq, self.n_agents, self.input_size)

        if not training:
            # Сброс hidden при is_init — из lstm.py:191-197
            h_0     = torch.where(expand_as_right(is_init, h_0),     0, h_0)
            c_0     = torch.where(expand_as_right(is_init, c_0),     0, c_0)
            h_hat_0 = torch.where(expand_as_right(is_init, h_hat_0), 0, h_hat_0)
            c_hat_0 = torch.where(expand_as_right(is_init, c_hat_0), 0, c_hat_0)
            is_init = is_init.unsqueeze(1)  # эмулируем seq-dim

        assert is_init.shape == (batch, seq, 1)
        # expand на агентов — из lstm.py:200
        is_init = is_init.unsqueeze(-2).expand(batch, seq, self.n_agents, 1)

        if training:
            # Из lstm.py:202-221: centralised+share_params не имеет agent-dim в hidden
            if self.centralised and self.share_params:
                shape       = (batch, self.n_layers, self.hidden_size)
                hyper_shape = (batch, self.n_layers, self.hyper_size)
            else:
                shape       = (batch, self.n_agents, self.n_layers, self.hidden_size)
                hyper_shape = (batch, self.n_agents, self.n_layers, self.hyper_size)
            h_0     = torch.zeros(shape,       device=self.device, dtype=torch.float)
            c_0     = h_0.clone()
            h_hat_0 = torch.zeros(hyper_shape, device=self.device, dtype=torch.float)
            c_hat_0 = h_hat_0.clone()

        if self.centralised:
            # Concat obs агентов вдоль feature-dim — из lstm.py:222-224
            input   = input.view(batch, seq, self.n_agents * self.input_size)
            is_init = is_init[..., 0, :]  # агентская dim не нужна для centralised

        output, h_n, c_n, h_hat_n, c_hat_n = self._run_net(input, is_init, h_0, c_0, h_hat_0, c_hat_0, global_state)

        if self.centralised and self.share_params:
            # Один выход на всех агентов → broadcast — из lstm.py:228-231
            output = output.unsqueeze(-2).expand(batch, seq, self.n_agents, self.hidden_size)

        if not training:
            output = output.squeeze(1)   # убираем seq-dim — из lstm.py:233-234
        if missing_batch:
            output  = output.squeeze(0)
            h_n     = h_n.squeeze(0)
            c_n     = c_n.squeeze(0)
            h_hat_n = h_hat_n.squeeze(0)
            c_hat_n = c_hat_n.squeeze(0)

        return output, h_n, c_n, h_hat_n, c_hat_n

    def _run_net(self, input, is_init, h_0, c_0, h_hat_0, c_hat_0, global_state=None):
        # Аналог run_net (lstm.py:241-266), расширен на h_hat/c_hat
        # vmap позволяет прогнать одну сеть (_empty_net) с разными весами для каждого агента
        # global_state не имеет agent-dim → in_dims=None (broadcast по агентам)
        if not self.share_params:
            # in_dims: (..., global_state=None) — global_state broadcast на агентов
            if self.centralised:
                in_dims  = (0, None, None, -3, -3, -3, -3, None)
            else:
                in_dims  = (0, -2, -2, -3, -3, -3, -3, None)
            out_dims = (-2, -3, -3, -3, -3)
            output, h_n, c_n, h_hat_n, c_hat_n = self._vmap_func_module(
                self._empty_net, in_dims, out_dims
            )(self.params, input, is_init, h_0, c_0, h_hat_0, c_hat_0, global_state)
        else:
            # share_params: веса одни, vmap только по agent dim входов — из lstm.py:255-264
            with self.params.to_module(self._empty_net):
                if self.centralised:
                    output, h_n, c_n, h_hat_n, c_hat_n = self._empty_net(
                        input, is_init, h_0, c_0, h_hat_0, c_hat_0, global_state
                    )
                else:
                    output, h_n, c_n, h_hat_n, c_hat_n = torch.vmap(
                        self._empty_net,
                        in_dims=(-2, -2, -3, -3, -3, -3, None),
                        out_dims=(-2, -3, -3, -3, -3),
                    )(input, is_init, h_0, c_0, h_hat_0, c_hat_0, global_state)

        return output, h_n, c_n, h_hat_n, c_hat_n

    @staticmethod
    def _vmap_func_module(module, in_dims, out_dims):
        # Из lstm.py:268-273 — без изменений
        def exec_module(params, *inputs):
            with params.to_module(module):
                return module(*inputs)
        return torch.vmap(exec_module, in_dims=in_dims, out_dims=out_dims)


class HyperLstm(Model):
    """Аналог Lstm (lstm.py:282) — drop-in замена с HyperLSTM вместо LSTM.

    Принимает те же параметры что и Lstm, плюс hyper_size и n_z.
    """

    def __init__(
        self,
        hidden_size: int,
        hyper_size: int,
        n_z: int,
        n_layers: int,
        bias: bool,
        compile: bool,
        hyper_use_global_state: bool = False,
        global_state_size: int = 0,
        **kwargs,
    ):
        # Из lstm.py:316-328: передаём все BenchMARL-параметры в базовый Model
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.hidden_size = hidden_size
        self.hyper_size  = hyper_size
        self.n_z         = n_z
        self.n_layers    = n_layers
        self.bias        = bias
        self.compile     = compile
        self.hyper_use_global_state = hyper_use_global_state
        self.global_state_size = global_state_size

        # 4 ключа hidden вместо 2 в lstm.py:330-337
        # h/c — основной LSTM, h_hat/c_hat — hyper-LSTM
        self.hidden_state_name_h     = (self.agent_group, f"_hidden_hyper_lstm_h_{self.model_index}")
        self.hidden_state_name_c     = (self.agent_group, f"_hidden_hyper_lstm_c_{self.model_index}")
        self.hidden_state_name_h_hat = (self.agent_group, f"_hidden_hyper_lstm_h_hat_{self.model_index}")
        self.hidden_state_name_c_hat = (self.agent_group, f"_hidden_hyper_lstm_c_hat_{self.model_index}")

        # Регистрируем все 4 как rnn_keys — из lstm.py:339-342
        self.rnn_keys = unravel_key_list([
            "is_init",
            self.hidden_state_name_h,
            self.hidden_state_name_c,
            self.hidden_state_name_h_hat,
            self.hidden_state_name_c_hat,
        ])
        self.in_keys += self.rnn_keys

        # Если гиперсеть использует global_state — добавляем GLOBAL_STATE_KEY в in_keys.
        # В input_spec его нет (он не идёт в main LSTM), поэтому добавляем только в in_keys.
        if self.hyper_use_global_state:
            self.in_keys.append(GLOBAL_STATE_KEY)

        # input_features: только obs-фичи (не global_state) — размерность входа main LSTM
        self.input_features  = sum([spec.shape[-1] for spec in self.input_spec.values(True, True)])
        self.output_features = self.output_leaf_spec.shape[-1]

        # Из lstm.py:355-382: actor → MultiAgent*, critic (global) → ModuleList
        if self.input_has_agent_dim:
            self.hyper_lstm = MultiAgentHyperLSTM(
                input_size=self.input_features,
                hidden_size=self.hidden_size,
                hyper_size=self.hyper_size,
                n_z=self.n_z,
                n_agents=self.n_agents,
                device=self.device,
                centralised=self.centralised,
                share_params=self.share_params,
                n_layers=self.n_layers,
                bias=self.bias,
                compile=self.compile,
                hyper_use_global_state=self.hyper_use_global_state,
                global_state_size=self.global_state_size,
            )
        else:
            # Глобальный вход (критик без agent-dim) — из lstm.py:368-382
            self.hyper_lstm = nn.ModuleList(
                [
                    _get_hyper_net(
                        self.input_features,
                        self.hidden_size,
                        self.hyper_size,
                        self.n_z,
                        self.n_layers,
                        self.bias,
                        self.device,
                        self.compile,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

        # MLP-голова после RNN — из lstm.py:384-410, без изменений
        mlp_net_kwargs = {
            "_".join(k.split("_")[1:]): v
            for k, v in kwargs.items()
            if k.startswith("mlp_")
        }
        if self.output_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=self.hidden_size,
                n_agent_outputs=self.output_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **mlp_net_kwargs,
            )
        else:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=self.hidden_size,
                        out_features=self.output_features,
                        device=self.device,
                        **mlp_net_kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )

    def _perform_checks(self):
        # Из lstm.py:412-444, заменены только строки сообщений об ошибках
        super()._perform_checks()

        input_shape = None
        for input_key, input_spec in self.input_spec.items(True, True):
            if (self.input_has_agent_dim and len(input_spec.shape) == 2) or (
                not self.input_has_agent_dim and len(input_spec.shape) == 1
            ):
                if input_shape is None:
                    input_shape = input_spec.shape[:-1]
                else:
                    if input_spec.shape[:-1] != input_shape:
                        raise ValueError(
                            f"HyperLSTM inputs should all have the same shape up to the last dimension, got {self.input_spec}"
                        )
            else:
                raise ValueError(
                    f"HyperLSTM input value {input_key} from {self.input_spec} has an invalid shape, maybe you need a CNN?"
                )
        if self.input_has_agent_dim:
            if input_shape[-1] != self.n_agents:
                raise ValueError(
                    "If the HyperLSTM input has the agent dimension,"
                    f" the second to last spec dimension should be the number of agents, got {self.input_spec}"
                )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the HyperLSTM output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Из lstm.py:446-505, расширен на h_hat/c_hat

        # Concat obs-входов для main LSTM: всё кроме rnn_keys и global_state
        # global_state (если hyper_use_global_state) идёт только в hyper-ветку
        input = torch.cat(
            [tensordict.get(in_key) for in_key in self.in_keys
             if in_key not in self.rnn_keys and in_key != GLOBAL_STATE_KEY],
            dim=-1,
        )
        global_state = (
            tensordict.get(GLOBAL_STATE_KEY) if self.hyper_use_global_state else None
        )

        # Читаем 4 hidden из tensordict — из lstm.py:456-458, расширено
        h_0     = tensordict.get(self.hidden_state_name_h,     None)
        c_0     = tensordict.get(self.hidden_state_name_c,     None)
        h_hat_0 = tensordict.get(self.hidden_state_name_h_hat, None)
        c_hat_0 = tensordict.get(self.hidden_state_name_c_hat, None)
        is_init = tensordict.get("is_init")

        training = h_0 is None  # из lstm.py:460

        # Из lstm.py:462-466: actor path (input_has_agent_dim=True)
        if self.input_has_agent_dim:
            output, h_n, c_n, h_hat_n, c_hat_n = self.hyper_lstm(
                input, is_init, h_0, c_0, h_hat_0, c_hat_0, global_state=global_state
            )
            if not self.output_has_agent_dim:
                output = output[..., 0, :]
        else:
            # Из lstm.py:467-487: critic path — глобальный вход, hidden сбрасывается каждый раз
            batch = input.shape[0]
            seq   = input.shape[1]
            assert input.shape == (batch, seq, self.input_features)
            assert is_init.shape == (batch, seq, 1)

            h_0     = torch.zeros((batch, self.n_layers, self.hidden_size), device=self.device, dtype=torch.float)
            c_0     = h_0.clone()
            h_hat_0 = torch.zeros((batch, self.n_layers, self.hyper_size),  device=self.device, dtype=torch.float)
            c_hat_0 = h_hat_0.clone()

            if self.share_params:
                output, _, _, _, _ = self.hyper_lstm[0](input, is_init, h_0, c_0, h_hat_0, c_hat_0)
            else:
                outputs = []
                for net in self.hyper_lstm:
                    out, _, _, _, _ = net(input, is_init, h_0, c_0, h_hat_0, c_hat_0)
                    outputs.append(out)
                output = torch.stack(outputs, dim=-2)

        # MLP-голова — из lstm.py:489-499, без изменений
        if self.output_has_agent_dim:
            output = self.mlp.forward(output)
        else:
            if not self.share_params:
                output = torch.stack([net(output) for net in self.mlp], dim=-2)
            else:
                output = self.mlp[0](output)

        tensordict.set(self.out_key, output)
        # Из lstm.py:502-504, расширено на h_hat/c_hat
        if not training:
            tensordict.set(("next", *self.hidden_state_name_h),     h_n)
            tensordict.set(("next", *self.hidden_state_name_c),     c_n)
            tensordict.set(("next", *self.hidden_state_name_h_hat), h_hat_n)
            tensordict.set(("next", *self.hidden_state_name_c_hat), c_hat_n)
        return tensordict


@dataclass
class HyperLstmConfig(ModelConfig):
    """Аналог LstmConfig (lstm.py:508), добавлены hyper_size и n_z."""

    hidden_size: int = MISSING
    hyper_size:  int = MISSING
    n_z:         int = MISSING
    n_layers:    int = MISSING
    bias:       bool = MISSING
    compile:    bool = MISSING

    mlp_num_cells:        Sequence[int]    = MISSING
    mlp_layer_class:      Type[nn.Module]  = MISSING
    mlp_activation_class: Type[nn.Module]  = MISSING

    mlp_activation_kwargs: Optional[dict]  = None
    mlp_norm_class:        Type[nn.Module] = None
    mlp_norm_kwargs:       Optional[dict]  = None

    # Флаг: подавать global_state [h; global_state] на вход hyper-ячейки вместо [h; x]
    hyper_use_global_state: bool = False
    global_state_size:       int = 0

    @staticmethod
    def associated_class():
        return HyperLstm

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        # Из lstm.py:534-545, добавлены h_hat и c_hat
        return Composite(
            {
                f"_hidden_hyper_lstm_h_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hidden_size)
                ),
                f"_hidden_hyper_lstm_c_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hidden_size)
                ),
                f"_hidden_hyper_lstm_h_hat_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hyper_size)
                ),
                f"_hidden_hyper_lstm_c_hat_{model_index}": Unbounded(
                    shape=(self.n_layers, self.hyper_size)
                ),
            }
        )
