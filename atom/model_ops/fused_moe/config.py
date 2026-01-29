import logging
from typing import Union, NamedTuple, ClassVar, TYPE_CHECKING
from dataclasses import dataclass
import torch

if TYPE_CHECKING:
    from atom.model_ops.moe import FusedMoEParallelConfig

logger = logging.getLogger("atom")


class _GroupShape(NamedTuple):
    row: int
    col: int


class GroupShape(_GroupShape):
    """
    This class describes the quantization group shape.
    It includes static members for common shapes (per-tensor, per-token).
    """

    # Aliases for common quantization group shapes
    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]

    def is_per_tensor(self) -> bool:
        return self.row == -1 and self.col == -1

    def is_per_token(self) -> bool:
        return self.row == 1 and self.col == -1

    def is_per_group(self) -> bool:
        return self.row == 1 and self.col >= 1


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)


def _quant_flags_to_group_shape(
    quant_dtype: torch.dtype | str | None,
    per_act_token_quant: bool,
    block_shape: list[int] | None,
) -> tuple[GroupShape | None, GroupShape | None]:
    """
    Convert MoE quantization flags into more generic GroupShapes.
    """
    a_shape: GroupShape | None
    w_shape: GroupShape | None
    if block_shape is not None:
        assert not per_act_token_quant
        # TODO(bnell): this is not quite right for activations since first
        # dim should be 1.
        a_shape = GroupShape(row=block_shape[0], col=block_shape[1])
        w_shape = GroupShape(row=block_shape[0], col=block_shape[1])
    else:
        w_shape = None
        a_shape = None if quant_dtype is None else GroupShape.PER_TENSOR

        if per_act_token_quant:
            a_shape = GroupShape.PER_TOKEN

    return a_shape, w_shape


@dataclass
class FusedMoEQuantDesc:
    """
    A quantization descriptor for fused MoE ops. This class can describe
    either activations or weights.
    """

    # The quantized type of this parameters.  None means unquantized or
    # already quantized.
    dtype: torch.dtype | str | None = None

    # A field that describes the quantization group shape, from quant_utils.py.
    #  * (-1, -1)   for per-tensor quantization
    #  * (1, -1)    for per-row quantization
    #  * (-1, 1)    for per-column quantization
    #  * (128, 128) for 128x128 deepseek style block quantization
    #  * (1, 128)   for deepseek style activation quantization
    #               (i.e. per-token-per-group)
    shape: GroupShape | None = None

    # Quantization scales.
    scale: Union[torch.Tensor, "PrecisionConfig", None] = None  # noqa: F821

    # Quantization alphas or gscales, used for nvfp4 types.
    alpha_or_gscale: torch.Tensor | None = None

    # Zero points for int4/int8 types
    zp: torch.Tensor | None = None

    # Biases for GPT triton MoE
    bias: torch.Tensor | None = None


@dataclass
class FusedMoEQuantConfig:
    """
    Simplified FusedMoEQuantConfig for MoE quantization parameters.
    Contains activation and weight quantization descriptors.
    """

    _a1: FusedMoEQuantDesc
    _a2: FusedMoEQuantDesc
    _w1: FusedMoEQuantDesc
    _w2: FusedMoEQuantDesc

    def __post_init__(self):
        assert (
            not self.per_act_token_quant or self.block_shape is None
        ), "illegal quantization"

    # === Core properties ===

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self._a1.dtype

    @property
    def is_per_act_token(self) -> bool:
        return self._a1.shape == GroupShape.PER_TOKEN

    @property
    def per_act_token_quant(self) -> bool:
        return self._a1.shape == GroupShape.PER_TOKEN

    @property
    def block_shape(self) -> list[int] | None:
        if (
            self._a1.shape is not None
            and self._a1.shape != GroupShape.PER_TENSOR
            and self._a1.shape != GroupShape.PER_TOKEN
        ):
            return [self._a1.shape.row, self._a1.shape.col]
        return None

    @property
    def is_block_quantized(self) -> bool:
        return self.block_shape is not None

    # === Scale/Bias accessors ===

    @property
    def a1_scale(self) -> torch.Tensor | None:
        return self._a1.scale if isinstance(self._a1.scale, torch.Tensor) else None

    @property
    def a2_scale(self) -> torch.Tensor | None:
        return self._a2.scale if isinstance(self._a2.scale, torch.Tensor) else None

    @property
    def w1_scale(self) -> torch.Tensor | None:
        return self._w1.scale if isinstance(self._w1.scale, torch.Tensor) else None

    @property
    def w2_scale(self) -> torch.Tensor | None:
        return self._w2.scale if isinstance(self._w2.scale, torch.Tensor) else None

    @property
    def w1_bias(self) -> torch.Tensor | None:
        return self._w1.bias

    @property
    def w2_bias(self) -> torch.Tensor | None:
        return self._w2.bias

    @staticmethod
    def make(
        quant_dtype: torch.dtype | str | None = None,
        per_act_token_quant: bool = False,
        block_shape: list[int] | None = None,
        w1_scale: Union[torch.Tensor, "PrecisionConfig", None] = None,  # noqa: F821
        w2_scale: Union[torch.Tensor, "PrecisionConfig", None] = None,  # noqa: F821
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        weight_dtype: torch.dtype | str | None = None,
    ) -> "FusedMoEQuantConfig":
        """Builder function for FusedMoEQuantConfig."""
        assert not isinstance(quant_dtype, str) or quant_dtype in {
            "nvfp4",
            "mxfp4",
            "mxfp6_e3m2",
            "mxfp6_e2m3",
        }
        assert not isinstance(weight_dtype, str) or weight_dtype in {
            "nvfp4",
            "mxfp4",
            "mxfp6_e3m2",
            "mxfp6_e2m3",
        }

        if weight_dtype is None:
            weight_dtype = quant_dtype

        a_shape, w_shape = _quant_flags_to_group_shape(quant_dtype, False, block_shape)
        quant_config = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(quant_dtype, a_shape, a1_scale),
            _a2=FusedMoEQuantDesc(quant_dtype, a_shape, a2_scale),
            _w1=FusedMoEQuantDesc(weight_dtype, w_shape, w1_scale, None, None, w1_bias),
            _w2=FusedMoEQuantDesc(weight_dtype, w_shape, w2_scale, None, None, w2_bias),
        )
        assert quant_config.per_act_token_quant == per_act_token_quant
        assert quant_config.block_shape == block_shape
        return quant_config


def biased_moe_quant_config(
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for unquantized activations with biases.
    """
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(bias=w1_bias),
        _w2=FusedMoEQuantDesc(bias=w2_bias),
    )


def mxfp4_w4a16_moe_quant_config(
    w1_scale: Union[torch.Tensor, "PrecisionConfig"],  # noqa: F821
    w2_scale: Union[torch.Tensor, "PrecisionConfig"],  # noqa: F821
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for unquantized activations and mxfp4 weights.
    """
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_scale, None, None, w1_bias),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_scale, None, None, w2_bias),
    )


def fp8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for fp8 activations and fp8 weights.
    """
    return FusedMoEQuantConfig.make(
        torch.float8_e4m3fn,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
    )


FUSED_MOE_UNQUANTIZED_CONFIG: FusedMoEQuantConfig = FusedMoEQuantConfig.make()


@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int

    num_local_experts: int
    moe_parallel_config: "FusedMoEParallelConfig"

    # The activation type.
    in_dtype: torch.dtype | str | None = None

    max_num_tokens: int = 256

    has_bias: bool = False

    is_act_and_mul: bool = True

    is_lora_enabled: bool = False

    def __post_init__(self):
        if self.dp_size > 1:
            logger.debug("Using FusedMoEConfig::max_num_tokens=%d", self.max_num_tokens)

        assert self.max_num_tokens > 0

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_mori_kernels(self):
        return self.moe_parallel_config.use_mori_kernels
