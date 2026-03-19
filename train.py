import sys
from typing import List, Optional, Tuple

from swift.llm.train import SwiftPt, SwiftRLHF, SwiftSft


def _pop_option(argv: List[str], name: str) -> Tuple[List[str], Optional[str]]:
    """Remove a CLI option from argv and return its value if present."""
    cleaned_argv: List[str] = []
    value: Optional[str] = None
    skip_next = False

    for idx, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if arg == name:
            if idx + 1 < len(argv):
                value = argv[idx + 1]
                skip_next = True
            continue

        prefix = f'{name}='
        if arg.startswith(prefix):
            value = arg[len(prefix):]
            continue

        cleaned_argv.append(arg)

    return cleaned_argv, value


def _peek_option(argv: List[str], name: str) -> Optional[str]:
    _, value = _pop_option(list(argv), name)
    return value


def _get_runner_cls(argv: List[str]):
    pipeline = _peek_option(argv, '--pipeline') or 'sft'

    if pipeline == 'sft':
        return SwiftSft
    if pipeline == 'pt':
        return SwiftPt
    if pipeline == 'rlhf':
        rlhf_type = _peek_option(argv, '--rlhf_type')
        if rlhf_type == 'opsd':
            from opsd import SwiftOPSD
            return SwiftOPSD
        return SwiftRLHF

    raise ValueError(f'Unsupported pipeline: {pipeline}. Expected one of: sft, rlhf, pt.')


def main(argv: Optional[List[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cleaned_argv, _ = _pop_option(argv, '--pipeline')
    runner_cls = _get_runner_cls(argv)
    return runner_cls(cleaned_argv).main()


if __name__ == '__main__':
    main()
