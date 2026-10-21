"""
Поиск актуального порта беспроводной отладки.

Пример:
    python tools/find_adb_port.py
    python tools/find_adb_port.py --ip 192.168.31.156 --start 30000 --end 50000

Порт Wi-Fi ADB меняется после перезагрузки. Скрипт сканирует диапазон,
пробует `adb connect` к каждому открытому порту и печатает рабочий адрес.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adb_client import AdbClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Поиск порта Wi-Fi ADB")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--start", type=int, default=30000)
    parser.add_argument("--end", type=int, default=50001)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        dev = yaml.safe_load(handle)["device"]

    ip = args.ip or dev["ip"]
    client = AdbClient(dev["adb_path"], ip, dev["port"], timeout=8)

    ports = client.scan_wireless_ports(start=args.start, end=args.end)
    if not ports:
        print("Открытых портов не найдено. Устройство, вероятно, спит или Wi-Fi выключен.")
        return 1

    for port in ports:
        address = f"{ip}:{port}"
        result = subprocess.run(
            [dev["adb_path"], "connect", address],
            capture_output=True,
            text=True,
        )
        output = f"{result.stdout} {result.stderr}".strip()
        devices = subprocess.run(
            [dev["adb_path"], "devices"], capture_output=True, text=True
        ).stdout
        online = f"{address}\tdevice" in devices
        print(f"{address}: {'ONLINE' if online else 'offline'} | {output}")
        if online:
            print(f"\nРабочий адрес: {address}")
            print(f"Укажите в config.yaml: port: {port}")
            return 0

    print("\nПорты открыты, но ни один не дал online. Разблокируйте телефон "
          "и подтвердите отладку по Wi-Fi.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
