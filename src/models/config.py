from dataclasses import dataclass


@dataclass
class ViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    use_rope: bool = True
    rope_base: float = 10000.0
    use_slots: bool = False
    num_slots: int = 7
    d_slot: int = 64
    slot_iterations: int = 3
    dropout: float = 0.0


@dataclass
class BERTConfig:
    vocab_size: int = 30522
    max_seq_len: int = 512
    num_classes: int = 2
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    use_rope: bool = True
    rope_base: float = 10000.0
    use_slots: bool = False
    num_slots: int = 7
    d_slot: int = 64
    slot_iterations: int = 3
    dropout: float = 0.0


@dataclass
class MultimodalFusionConfig:
    vit_d_model: int = 768
    bert_d_model: int = 768
    joint_d_model: int = 512
    num_heads: int = 8
    num_layers: int = 2
    mlp_ratio: float = 4.0
    use_rope: bool = True
    rope_base: float = 10000.0
    max_seq_len: int = 2048
    dropout: float = 0.1