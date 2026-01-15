from memif import MemIfBase


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


class LRUCacheMemIf(MemIfBase):
    def reset(self):
        cfg = self.config or {}
        self.size_bytes = _parse_int(cfg.get("size_bytes", 2048), 2048)
        self.ways = _parse_int(cfg.get("ways", 2), 2)
        self.line_bytes = _parse_int(cfg.get("line_bytes", 32), 32)
        self.hit_penalty = _parse_int(cfg.get("hit_penalty", 0), 0)
        self.miss_penalty = _parse_int(cfg.get("miss_penalty", 0), 0)
        self.write_allocate = _parse_bool(cfg.get("write_allocate", True), True)

        denom = self.line_bytes * self.ways
        if self.size_bytes <= 0 or denom <= 0:
            raise ValueError("Cache size, line size, and ways must be positive")
        if self.size_bytes % denom != 0:
            raise ValueError("Cache size must be a multiple of line_bytes * ways")
        self.num_sets = self.size_bytes // denom
        if self.num_sets <= 0:
            raise ValueError("Cache configuration yields zero sets")
        self.sets = [[] for _ in range(self.num_sets)]
        self.accesses = 0
        self.hits = 0
        self.misses = 0

    def _access_line(self, line_addr, is_write):
        set_idx = line_addr % self.num_sets
        tag = line_addr // self.num_sets
        lines = self.sets[set_idx]
        self.accesses += 1
        if tag in lines:
            self.hits += 1
            lines.remove(tag)
            lines.insert(0, tag)
            return self.hit_penalty
        self.misses += 1
        if not is_write or self.write_allocate:
            if len(lines) >= self.ways:
                lines.pop()
            lines.insert(0, tag)
        return self.miss_penalty

    def access(self, addr, size, is_write=False):
        if size <= 0:
            return 0
        start_line = addr // self.line_bytes
        end_line = (addr + size - 1) // self.line_bytes
        penalty = 0
        for line_addr in range(start_line, end_line + 1):
            penalty += self._access_line(line_addr, is_write)
        return penalty

    def stats(self):
        return {"accesses": self.accesses, "hits": self.hits, "misses": self.misses}
