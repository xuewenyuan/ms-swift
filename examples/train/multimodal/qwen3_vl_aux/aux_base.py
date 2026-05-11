from typing import Any, Dict, Optional

from torch import nn


class BaseModule(nn.Module):

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.cfg = cfg or {}

