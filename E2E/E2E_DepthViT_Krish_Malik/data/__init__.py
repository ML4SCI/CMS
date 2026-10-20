"""DepthViT modular data layer.

Importing this package registers all available data formats.
Add a new dataset:
    1. Create data/<name>.py implementing a DatasetAdapter subclass
       and a @register_format("<name>") wrapper.
    2. Add `from . import <name>` below.
    3. Set cfg["data"]["format"] = "<name>" in the JSON config.
"""

from .base import (
    DatasetAdapter,
    ScientificIterableDataset,
    FORMAT_REGISTRY,
    register_format,
    make_loaders_dispatch,
    make_loaders_for_adapters,
)

                                                                
                                                               
                                                                      

try:
    from . import lhc_jets              
except ImportError as _e:
    print(f"[data] lhc_jets unavailable: {_e}")

try:
    from . import imagenet_wds              
except ImportError as _e:
    print(f"[data] imagenet_wds unavailable: {_e}")

try:
    from . import imagewoof              
except ImportError as _e:
    print(f"[data] imagewoof unavailable: {_e}")

                                             
      
                           
                           
                                                
      
                               
                           
                                                    
      
                           
                           
                                                

__all__ = [
    "DatasetAdapter",
    "ScientificIterableDataset",
    "FORMAT_REGISTRY",
    "register_format",
    "make_loaders_dispatch",
    "make_loaders_for_adapters",
]
