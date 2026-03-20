from dataclasses import dataclass
from typing import List, Optional

from swift.llm.train import SwiftPt, SwiftRLHF, SwiftSft
from swift.utils import parse_args


@dataclass
class RouterArguments:
    rlhf_type: Optional[str] = None


def _get_runner_cls(argv: Optional[List[str]] = None):
    router_args, _ = parse_args(RouterArguments, argv)
    rlhf_type = router_args.rlhf_type

    if rlhf_type is not None:
        if rlhf_type == 'opsd':
            from opsd import SwiftOPSD
            return SwiftOPSD
        return SwiftRLHF

    return SwiftSft


def main(argv: Optional[List[str]] = None):
    runner_cls = _get_runner_cls(argv)
    return runner_cls(argv).main()


if __name__ == '__main__':
    main()
