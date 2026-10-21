#!/usr/bin/env python3
"""Read-only ADB check: list one authorized phone and save a screenshot."""
import shutil, subprocess, sys, time
from pathlib import Path

def adb(*args):
    exe = shutil.which("adb")
    if not exe:
        raise RuntimeError("adb not found; install Android Platform Tools and add adb to PATH")
    p = subprocess.run([exe, *args], capture_output=True, timeout=15)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return p.stdout

def main():
    try:
        listing = adb("devices").decode(errors="replace")
        print(listing)
        devices = [x.split()[0] for x in listing.splitlines()[1:]
                   if len(x.split()) >= 2 and x.split()[1] == "device"]
        if len(devices) != 1:
            print("Expected exactly one authorized device with status 'device'. Disconnect extras or authorize phone.", file=sys.stderr)
            return 2
        serial = devices[0]
        size = adb("-s", serial, "shell", "wm", "size").decode(errors="replace").strip()
        density = adb("-s", serial, "shell", "wm", "density").decode(errors="replace").strip()
        png = adb("-s", serial, "exec-out", "screencap", "-p")
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("screencap did not return PNG data")
        out = Path("artifacts"); out.mkdir(exist_ok=True)
        target = out / f"phone-screen-{time.strftime('%Y%m%d-%H%M%S')}.png"
        target.write_bytes(png)
        print("Device:", serial, "|", size, "|", density)
        print("Screenshot:", target.resolve())
        print("Success: no taps or state-changing commands were issued.")
        return 0
    except Exception as e:
        print("ADB smoke test failed:", e, file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
