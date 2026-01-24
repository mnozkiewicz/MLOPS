from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Time Server")

@mcp.tool(description="Returns current date in the format 'Year-Month-Day' (YYYY-MM-DD)")
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

@mcp.tool(description="Returns current date and time in ISO 8601 format (up to seconds)")
def get_current_datetime() -> str:
    return datetime.now().isoformat(timespec='seconds')


mcp.run(transport="streamable-http", port=8002)