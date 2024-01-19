"""Distribution and Architecture utilities"""

import platform


def architecture_is_apple_arm() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"
