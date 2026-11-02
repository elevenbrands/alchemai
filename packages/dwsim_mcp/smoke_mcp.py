"""Start the AlchemAI MCP server over stdio, list tools, build and solve a pump through MCP."""
import asyncio, json, os, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]

def _text(r): return "\n".join(c.text for c in r.content if getattr(c, "type", "") == "text")

async def main():
    params = StdioServerParameters(command="uv", args=["run", "--directory", str(ROOT), "python", "-m", "packages.dwsim_mcp.server"], env=dict(os.environ))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print(len(tools.tools), "tools:", sorted(t.name for t in tools.tools))
            fs = json.loads(_text(await s.call_tool("new_flowsheet", {"name": "p", "compounds": ["Water"], "property_package": "SteamTables"})))["flowsheet_id"]
            await s.call_tool("add_material_stream", {"flowsheet_id": fs, "name": "w", "temperature_K": 298.15, "pressure_Pa": 1e5, "mass_flow_kg_s": 10.0, "mole_fractions": {"Water": 1.0}})
            await s.call_tool("add_material_stream", {"flowsheet_id": fs, "name": "hp"})
            await s.call_tool("add_unit", {"flowsheet_id": fs, "kind": "pump", "name": "P-1", "params": {"outlet_P_Pa": 11e5, "efficiency_pct": 70}})
            await s.call_tool("connect", {"flowsheet_id": fs, "unit": "P-1", "feeds": ["w"], "products": ["hp"]})
            sol = json.loads(_text(await s.call_tool("solve", {"flowsheet_id": fs})))
            u = json.loads(_text(await s.call_tool("read_unit", {"flowsheet_id": fs, "name": "P-1"})))
            print("solve ok:", sol["ok"], " pump power kW:", round(u["power_kW"], 3))

asyncio.run(main())
