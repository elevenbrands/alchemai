"""Smoke test: start DWSIM's MCP server over stdio, list its tools, solve a minimal flowsheet.

Usage:
    uv run python packages/dwsim_client/smoke.py [path/to/dwsim-mcp.dll]

The server is DWSIM 10 (GPL-3.0) built from ~/tools/dwsim10:
    dotnet build tools/DWSIM.MCPServer/DWSIM.MCPServer.csproj -c Release
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_DLL = Path.home() / "tools/dwsim10/tools/DWSIM.MCPServer/bin/Release/net10.0/dwsim-mcp.dll"
DOTNET = Path.home() / ".dotnet/dotnet"


def _text(result) -> str:
    parts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    return "\n".join(parts)


async def main(dll: Path) -> None:
    if not dll.exists():
        sys.exit(f"MCP server not built: {dll}\nRun: cd ~/tools/dwsim10 && dotnet build "
                 "tools/DWSIM.MCPServer/DWSIM.MCPServer.csproj -c Release")
    params = StdioServerParameters(
        command=str(DOTNET), args=[str(dll), "--stdio"],
        env={**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"{len(names)} tools")
            for n in names:
                print(" ", n)

            # Minimal flowsheet smoke: create, inspect degrees of freedom, summary.
            r = await session.call_tool("dwsim_flowsheet_create", {})
            print("\n[create]", _text(r)[:400])
            r = await session.call_tool("dwsim_flowsheet_summary", {})
            print("[summary]", _text(r)[:400])


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DLL))
