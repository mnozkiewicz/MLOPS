import asyncio
import json
from contextlib import AsyncExitStack
from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from dotenv import load_dotenv
import os

load_dotenv()

class MCPManager:
    def __init__(self, servers: dict[str, str]):
        self.servers = servers
        self.clients = {}
        self.tools = []  # in OpenAI format
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        for url in self.servers.values():
            # initialize MCP session with Streamable HTTP client
            read, write, session_id = await self._stack.enter_async_context(
                streamable_http_client(url)
            )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            # use /list_tools MCP endpoint to get tools
            # parse each one to get OpenAI-compatible schema
            tools_resp = await session.list_tools()
            for t in tools_resp.tools:
                self.clients[t.name] = session
                self.tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema,
                        },
                    }
                )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()

    async def call_tool(self, name: str, args: dict) -> dict | str:
        # call the MCP tool with given arguments
        result = await self.clients[name].call_tool(name, arguments=args)
        return result.content[0].text


async def make_llm_request(messages: list[dict[str, str]]) -> str:

    tavily_api_key = os.environ["TAVILY_API_KEY"]
    mcp_servers = {
        "weather_forecast": "http://localhost:8001/mcp",
        "websearch": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}"
    }

    vllm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:11434/v1")

    async with MCPManager(mcp_servers) as mcp:
        
        for _ in range(10):
            response = vllm_client.chat.completions.create(
                model="qwen3:1.7b",
                messages=messages,
                tools=mcp.tools,
                tool_choice="auto",
                max_completion_tokens=200
            )

            response = response.choices[0].message

            if not response.tool_calls:
                messages.append({"role": response.role, "content": response.content})
                break

            messages.append(response)

            for tool_call in response.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"Executing tool '{func_name}'")

                func_result = await mcp.call_tool(func_name, func_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(func_result),
                    }
                )

    return messages


def main() -> None:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use tools if you need to."
            ),
        },
    ]

    last_message = 0

    while True:

        print("User: ", end="")
        prompt = input()
        if prompt == "exit":
            break
        
        user_prompt = {"role": "user", "content": f"{prompt} /no_think"}
        messages.append(user_prompt)
        messages = asyncio.run(make_llm_request(messages))

        for i in range(last_message+1, len(messages)):
            message = messages[i]
            if isinstance(message, dict) and message["role"] == "assistant" and message["content"]:
                content = message["content"]
                last_message = i
                print(f"AI assistant: {content}")



if __name__ == "__main__":
    main()