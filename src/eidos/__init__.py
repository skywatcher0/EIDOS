"""EIDOS: symbolic regression with isolated evidence and portable model records."""
__version__ = "0.2.0"

def __getattr__(name):
    if name == "EIDOSRegressor":
        from .estimator import EIDOSRegressor
        return EIDOSRegressor
    raise AttributeError(name)

__all__ = ["EIDOSRegressor", "__version__"]
