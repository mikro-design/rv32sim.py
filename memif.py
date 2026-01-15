import importlib


def _parse_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return int(value, 0)
    return int(value)

def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    return bool(value)


class MemIfBase:
    def __init__(self, read_cb, write_cb, config=None, name="memif"):
        self.read_cb = read_cb
        self.write_cb = write_cb
        self.name = name
        self.config = config or {}
        self.reset()

    def reset(self):
        pass

    def access(self, addr, size, is_write=False):
        return 0

    def stats(self):
        return {}


class FlatMemIf(MemIfBase):
    def reset(self):
        self.accesses = 0
        self.penalty = _parse_int(self.config.get("penalty", 0), 0)
        self.count_misses = _parse_bool(self.config.get("count_misses", False), False)

    def access(self, addr, size, is_write=False):
        self.accesses += 1
        return self.penalty

    def stats(self):
        stats = {"accesses": self.accesses}
        if self.count_misses:
            stats["hits"] = 0
            stats["misses"] = self.accesses
        return stats


def load_memif(spec, read_cb, write_cb, config=None, name="memif"):
    if not spec or ":" not in spec:
        raise ValueError(f"Invalid memif spec: {spec!r}")
    module_name, cls_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(read_cb, write_cb, config=config or {}, name=name)
