"""VoxControlResearch — uncertainty-aware voice intent recognition and adaptive command execution."""
__version__ = "1.0.0"
SOFTWARE_NAME = "VoxControlResearch"


def __getattr__(name):
    if name == "VoxModel":
        from .api import VoxModel
        return VoxModel
    raise AttributeError(name)
