import os
import json

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Initialize client
client = InferenceClient(token=os.environ["HF_TOKEN"], provider="nebius")

user_message = "What's the weather like in San Francisco?"

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to weather data.",
    },
    {"role": "user", "content": user_message},
]


# Define multiple functions
def get_current_weather(location: str) -> dict:
    """Get current weather for a location."""
    return {"location": location, "temperature": "22°C", "condition": "Sunny"}


def get_weather_forecast(location: str, date: str) -> dict:
    """Get weather forecast for a location."""
    return {
        "location": location,
        "date": date,
        "forecast": "Sunny with chance of rain",
        "temperature": "20°C",
    }


# Function registry
AVAILABLE_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast,
}

# Multiple tool schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get weather forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format",
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]


# Use strict mode to ensure function calls follow your schema exactly. This is useful
# to prevent the model from calling functions with unexpected arguments,
# or calling functions that don’t exist.
# # Define tools with strict mode
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_weather",
#             "description": "Get current weather for a location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {"type": "string", "description": "City name"},
#                 },
#                 "required": ["location"],
# +                "additionalProperties": False,  # Strict mode requirement
#             },
# +            "strict": True,  # Enable strict mode
#         },
#     },
# ]


response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # "auto": model decides when to call functions; tool_choice="required", must call at least one function
)


# execute the response
response_message = response.choices[0].message

# check if the model wants to call functions
if response_message.tool_calls:
    # process the tool calls
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        # execute the function
        if function_name == "get_current_weather":
            result = get_current_weather(function_args["location"])
        elif function_name == "get_weather_forecast":
            result = get_weather_forecast(
                function_args["location"], function_args["date"]
            )

        # add the result to the conversation
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(result),
            }
        )

    # get the final response with function results
    final_response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-0528",
        messages=messages,
    )

    print(final_response.choices[0].message.content)
else:
    print(response_message.content)
