"""Entry point of the packaged application."""
import sys

from voxcontrol.paths import configure_caches

configure_caches()

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        from voxcontrol.selftest import run
        sys.exit(run())
    from voxcontrol.gui.app import main
    main()
