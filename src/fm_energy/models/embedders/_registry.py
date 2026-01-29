import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("building-embedders")


__EMBEDDER_REGISTRY__: dict[str, type] = {}


def register_embedder(name: str):
    def wrapper(cls):
        if name in __EMBEDDER_REGISTRY__:
            raise ValueError(f"embedder {name} already registered.")
        else:
            __EMBEDDER_REGISTRY__[name] = cls
        return cls

    return wrapper


def get_embedder(name: str, **kwargs) -> object:
    embedder = __EMBEDDER_REGISTRY__.get(name, None)
    if embedder is None:
        raise ValueError(f"embedder {name} not registered.")
    return embedder(**kwargs)
