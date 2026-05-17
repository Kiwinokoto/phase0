from __future__ import annotations

import os
import sys

# Keep Pygame quiet and avoid accidental OpenGL/GLX paths on Linux laptops/VMs
# where Mesa DRI drivers may be missing. This must happen before importing pygame.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
if os.environ.get("ALIFE_FORCE_SOFTWARE_RENDER", "1") != "0":
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


def _warn_if_wrong_python() -> None:
    if sys.version_info < (3, 11):
        print(
            "WARNING: Python 3.11+ is recommended for this project.\n"
            f"Current interpreter: {sys.version.split()[0]} at {sys.executable}\n"
            "Use: ./run.sh\n"
            "or: source ./create_or_activate_env.sh && python run.py\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    _warn_if_wrong_python()
    from alife.viewer import main

    main()
