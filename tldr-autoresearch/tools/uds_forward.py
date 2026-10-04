#!/usr/bin/env python3
"""Forward 127.0.0.1:<port> (inside the agent sandbox's private network namespace) to a Unix socket.

The sandbox has no network except loopback; the only services the agent can reach are the ones we
forward explicitly (gateway for LLM calls, runner for training runs).

  uds_forward.py PORT:SOCKET [PORT:SOCKET ...]
"""
import asyncio
import sys


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(1 << 16)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def handler(sock_path):
    async def handle(client_r, client_w):
        try:
            up_r, up_w = await asyncio.open_unix_connection(sock_path)
        except OSError:
            client_w.close()
            return
        await asyncio.gather(pipe(client_r, up_w), pipe(up_r, client_w))
    return handle


async def main(specs):
    servers = []
    for spec in specs:
        port, path = spec.split(":", 1)
        servers.append(await asyncio.start_server(handler(path), "127.0.0.1", int(port)))
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
