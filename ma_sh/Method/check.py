import torch

def checkFormat(data: torch.Tensor, dtype=torch.float32, device: str='cpu', shape=None) -> bool:
    if data.dtype != dtype:
        print('[WARN][check::checkFormat]')
        print('\t dtype not matched!')
        print('\t', data.dtype, '!=', dtype)
        return False

    if data.device.type != device:
        print('[WARN][check::checkFormat]')
        print('\t device not matched!')
        print('\t', data.device, '!=', device)
        return False

    if shape is not None:
        if len(data.shape) != len(shape):
            print('[WARN][check::checkFormat]')
            print('\t shape channel not matched!')
            print('\t', data.shape, '!=', shape)
            return False

        for i in range(len(data.shape)):
            if data.shape[i] != shape[i]:
                print('[WARN][check::checkFormat]')
                print('\t shape dim not matched!')
                print('\t', data.shape, '!=', shape)
                return False

    return True
