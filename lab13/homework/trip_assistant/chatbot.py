import asyncio
import json
from openai import OpenAI
from .mcp_manager import MCPManager
from guardrails import Guard
from guardrails.hub import ProfanityFree
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="guardrails")

load_dotenv()
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

async def main() -> None:
    mcp_servers = {
        "weather": "http://localhost:8001/mcp", 
        "tavily": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
    }

    vllm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    profanity_guard = Guard().use(
        ProfanityFree, 
        on_fail="exception"
    )

    async with MCPManager(mcp_servers) as mcp:
        
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant. Use tools to plan trips."},
        ]

        print("\nWelcome to Trip planner")

        while True:
            user_input = await asyncio.to_thread(input, "\nUser: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            try:
                await asyncio.to_thread(profanity_guard.validate, user_input)
            except Exception:
                print(f"\nAI assistant: Profanity is not allowed in this chat")
                continue

            messages.append({"role": "user", "content": user_input})

            for _ in range(10): 
                completion = vllm_client.chat.completions.create(
                    model="",
                    messages=messages,
                    tools=mcp.tools,
                    tool_choice="auto",
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

                message = completion.choices[0].message
                msg_dict = {"role": message.role, "content": message.content}
                
                if message.tool_calls:
                    msg_dict["tool_calls"] = [
                        t.model_dump() for t in message.tool_calls
                    ]
                
                messages.append(msg_dict)
                if not message.tool_calls:
                    print(f"AI assistant: {message.content}")
                    break

                print("(Thinking...)")
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    print(f"Calling Tool: {func_name}({func_args})")
                    
                    tool_result = await mcp.call_tool(func_name, func_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(tool_result)
                    })

if __name__ == "__main__":
    asyncio.run(main())