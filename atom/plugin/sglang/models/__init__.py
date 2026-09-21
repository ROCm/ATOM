from atom.plugin.sglang.models.minimax_m3_processor import (
    register_minimax_m3_text_only_processor,
)
from atom.plugin.sglang.patches.qwen4_exp_recognition_patch import (
    register_qwen4_exp_processor,
)

register_minimax_m3_text_only_processor()
register_qwen4_exp_processor()
