import torch
import torch.nn as nn

from typing import Union, Optional

class BaseEncoder(nn.Module):
    """
    A wrapper around a given neural network that extracts
    activations from a specified layer during forward pass
    - using forward hooks
    """
    def __init__(self, net: nn.Module, layer: Union[str, int] = -2):
        """
        net: a neural network
        layer: the layer from which to extract the hidden representation
        """
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self._register_hook()
 
    def _find_layer(self) -> Optional[nn.Module]:
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None) # returns None if layer not found
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None
    
    def _register_hook(self) -> None:
        """
        General guidelines to register a forward hook:
        1. Define a hook function that takes three arguments: module, input, output
            1.a don't pass "self" as pytorch expects these three arguments
        2. Get the layer from the network
        3. Register the hook and use the output as intended
        4. Input can be used/modified with pre_forward_hook
        """
        def hook(_, __, output: torch.Tensor) -> None:
            self.hidden = output

        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        self.hook_handle = layer.register_forward_hook(hook)

    def remove_hook(self) -> None:
        # remove the hook when done
        
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

  
    def forward(self, x) -> torch.Tensor:
        if self.layer == -1:
            return self.net(x)
        
        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer ({self.layer}) not found'
        return hidden