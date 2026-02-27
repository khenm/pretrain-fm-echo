from typing import Dict, Type, Any, Optional

_MODELS: Dict[str, Type[Any]] = {}
_DATASETS: Dict[str, Type[Any]] = {}
_LOSSES: Dict[str, Type[Any]] = {}
_TTA_COMPONENTS: Dict[str, Type[Any]] = {}

def register_model(name: str):
    """
    Decorator to register a model class.
    """
    def decorator(cls: Type[Any]):
        _MODELS[name] = cls
        return cls
    return decorator

def register_dataset(name: str):
    """
    Decorator to register a dataset class.
    """
    def decorator(cls: Type[Any]):
        _DATASETS[name] = cls
        return cls
    return decorator

def register_loss(name: str):
    """
    Decorator to register a loss class.
    """
    def decorator(cls: Type[Any]):
        _LOSSES[name] = cls
        return cls
    return decorator

def register_tta_component(name: str):
    """
    Decorator to register a TTA component (Auditor, Calibrator, etc.).
    """
    def decorator(cls: Type[Any]):
        _TTA_COMPONENTS[name] = cls
        return cls
    return decorator

def get_model_class(name: str) -> Type[Any]:
    if name not in _MODELS:
        raise KeyError(f"Model '{name}' not found in registry. Available: {list(_MODELS.keys())}")
    return _MODELS[name]

def get_dataset_class(name: str) -> Type[Any]:
    if name not in _DATASETS:
        raise KeyError(f"Dataset '{name}' not found in registry. Available: {list(_DATASETS.keys())}")
    return _DATASETS[name]

def get_loss_class(name: str) -> Type[Any]:
    if name not in _LOSSES:
        raise KeyError(f"Loss '{name}' not found in registry. Available: {list(_LOSSES.keys())}")
    return _LOSSES[name]

def get_tta_component_class(name: str) -> Type[Any]:
    if name not in _TTA_COMPONENTS:
        raise KeyError(f"TTA Component '{name}' not found in registry. Available: {list(_TTA_COMPONENTS.keys())}")
    return _TTA_COMPONENTS[name]

def build_model(cfg: Dict[str, Any], device: Any) -> Any:
    """
    Instantiates a model based on the configuration.
    """
    model_name = cfg['model'].get('name', 'VAEUNet') # Default fallback
    model_cls = get_model_class(model_name)
    
    if hasattr(model_cls, 'from_config'):
        return model_cls.from_config(cfg).to(device)
    else:
        raise NotImplementedError(f"Model '{model_name}' does not implement 'from_config'.")

def get_dataloaders(cfg: Dict[str, Any]) -> Any:
    """
    Instantiate dataloaders based on the configuration.
    """
    data_name = cfg['data'].get('name', 'CAMUS').upper()
    dataset_cls = get_dataset_class(data_name)
    
    if hasattr(dataset_cls, 'get_dataloaders'):
        return dataset_cls.get_dataloaders(cfg)
    else:
         raise NotImplementedError(f"Dataset '{data_name}' does not implement 'get_dataloaders'.")

def build_loss(name: str, **kwargs) -> Any:
    """
    Instantiates a loss function from the registry.
    """
    loss_cls = get_loss_class(name)
    return loss_cls(**kwargs)
