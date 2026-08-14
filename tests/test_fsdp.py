from ds_distributed.fsdp_checkpoint import save_fsdp_checkpoint
from models import GPTModel, TransformerConfig


def create_model():
    config = TransformerConfig(
        vocab_size=100,
        max_seq_len=32,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
    )

    return GPTModel(config)


# def test_fsdp_checkpoint_requires_process_group(tmp_path):
#     if torch.distributed.is_initialized():
#         return

#     model = create_model()


#     try:
#         save_fsdp_checkpoint(
#             model,
#             tmp_path / "checkpoint.pt",
#         )
#     except AttributeError:
#         pass
#     else:
#         raise AssertionError(
#             "Expected FSDP checkpointing to require an FSDP model"
#         )
def test_fsdp_checkpoint_requires_fsdp_model(tmp_path):
    model = create_model()

    try:
        save_fsdp_checkpoint(
            model,
            tmp_path / "checkpoint.pt",
        )
    except TypeError as exc:
        assert "FullyShardedDataParallel" in str(exc)
    else:
        raise AssertionError("Expected FSDP checkpointing to require an FSDP model")
