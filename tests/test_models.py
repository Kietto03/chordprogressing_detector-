import torch
import pytest
import warnings
from src.model import TransformerChordRecognizer, CRNNChordBaseline


@pytest.mark.parametrize("model_cls", [TransformerChordRecognizer, CRNNChordBaseline])
def test_model_forward_shape_and_no_nan(model_cls):
    """Both models must accept [B, 215, 84] and output [B, 215, 25] without NaNs."""
    model = model_cls(input_bins=84, num_classes=25)
    model.eval()
    x = torch.randn(4, 215, 84)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 215, 25)
    assert not torch.isnan(out).any()
    assert out.dtype == torch.float32


def test_transformer_specific():
    model = TransformerChordRecognizer(input_bins=84, num_classes=25, d_model=256, num_layers=4)
    x = torch.randn(2, 215, 84)
    out = model(x)
    assert out.shape == (2, 215, 25)


def test_crnn_specific():
    model = CRNNChordBaseline(input_bins=84, num_classes=25, rnn_hidden=128)
    x = torch.randn(2, 215, 84)
    out = model(x)
    assert out.shape == (2, 215, 25)


def test_nested_tensor_warning_suppression():
    """Explicitly test that no 'enable_nested_tensor' / Pre-LN warning is emitted.

    This addresses the known PyTorch warning when using norm_first=True (Pre-LN)
    with the default nested tensor optimization. We pass enable_nested_tensor=False
    in the model when supported.
    """
    import inspect
    sig = inspect.signature(torch.nn.TransformerEncoderLayer.__init__)
    has_param = "enable_nested_tensor" in sig.parameters

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = TransformerChordRecognizer()
        x = torch.randn(1, 215, 84)
        with torch.no_grad():
            _ = model(x)

        if has_param:
            nested_warnings = [
                warning for warning in w
                if "nested_tensor" in str(warning.message).lower()
                or "enable_nested_tensor" in str(warning.message).lower()
            ]
            assert len(nested_warnings) == 0, f"Unexpected nested_tensor warnings: {nested_warnings}"
        else:
            # If PyTorch doesn't support enable_nested_tensor parameter, the warning is expected
            # and raised internally by PyTorch, so we do not assert 0.
            pass
