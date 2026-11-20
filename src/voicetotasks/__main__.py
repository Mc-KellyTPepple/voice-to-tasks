import sys

from .server import build_app, serve

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "serve":
        serve(build_app())
    elif cmd == "bench":
        from .bench import main
        main(sys.argv[2:])
    else:
        sys.exit("usage: python -m voicetotasks [serve|bench]")
