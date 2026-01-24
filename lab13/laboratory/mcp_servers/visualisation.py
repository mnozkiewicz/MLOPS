from fastmcp import FastMCP
from typing import Annotated, Optional
import matplotlib
import matplotlib.pyplot as plt
import io
import base64


matplotlib.use('Agg')
mcp = FastMCP("Visualization Server")

@mcp.tool(description="Generates a line plot from data and returns a base64 encoded PNG image.")
def line_plot(
    data: Annotated[list[list[float]], "One or more lists of numbers to plot. Example: [[1, 2, 3], [4, 5, 6]]"],
    title: Annotated[Optional[str], "The title of the plot"] = "Line Plot",
    x_label: Annotated[Optional[str], "Label for the X-axis"] = "X",
    y_label: Annotated[Optional[str], "Label for the Y-axis"] = "Y",
    legend: Annotated[bool, "Whether to show the legend"] = True,
) -> str:


    fig, ax = plt.subplots()

    for i, series in enumerate(data):
        ax.plot(series, label=f"Series {i+1}")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if legend and len(data) > 0:
        ax.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig) # Clean up memory
    
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    
    return img_str


mcp.run(transport="streamable-http", port=8003)