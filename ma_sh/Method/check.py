import torch

from ma_sh.Config.mode import DEBUG


def checkShape(source_shape, target_shape):
    if len(source_shape) != len(target_shape):
        print("[WARN][check::checkShape]")
        print("\t shape channel not matched!")
        print("\t", source_shape, "!=", target_shape)
        return False

    for i in range(len(source_shape)):
        if source_shape[i] != target_shape[i]:
            print("[WARN][check::checkShape]")
            print("\t shape dim not matched!")
            print("\t", source_shape, "!=", target_shape)
            return False

    return True


def checkFormat(
    data: torch.Tensor,
    dtype=torch.float32,
    device: str = "cpu",
    shape=None,
    have_grad=None,
) -> bool:
    if not DEBUG:
        return True

    if data.dtype != dtype:
        print("[WARN][check::checkFormat]")
        print("\t dtype not matched!")
        print("\t", data.dtype, "!=", dtype)
        return False

    if str(data.device) != device:
        print("[WARN][check::checkFormat]")
        print("\t device not matched!")
        print("\t", data.device, "!=", device)
        return False

    if shape is not None:
        if not checkShape(data.shape, shape):
            print("[WARN][check::checkFormat]")
            print("\t checkShape matched!")
            return False

    if have_grad is not None:
        grad_state = data.grad_fn is not None

        if grad_state != have_grad:
            print("[WARN][check::checkFormat]")
            print("\t grad state not matched!")
            print("\t", grad_state, "!=", have_grad)
            return False

    return True
