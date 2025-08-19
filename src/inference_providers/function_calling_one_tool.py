import os
import json

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
# Initialize client
client = InferenceClient(token=os.environ["HF_TOKEN"], provider="nebius")


# Define the function
def get_current_weather(location: str) -> dict:
    """Get weather information for a location."""
    # In production, this would call a real weather API
    weather_data = {
        "San Francisco": {"temperature": "22°C", "condition": "Sunny"},
        "New York": {"temperature": "18°C", "condition": "Cloudy"},
        "London": {"temperature": "15°C", "condition": "Rainy"},
    }

    return weather_data.get(
        location, {"location": location, "error": "Weather data not available"}
    )


# Define the function schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    },
]

user_message = "What's the weather like in San Francisco?"

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant with access to weather data.",
    },
    {"role": "user", "content": user_message},
]

# Initial API call with tools
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # Let the model decide when to call functions
)

# execute the response
response_message = response.choices[0].message
print(response_message)

# Check if model wants to call functions
if response_message.tool_calls:
    # Add assistant's response to messages
    messages.append(response_message)

    # Process each tool call
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        # Execute the function
        if function_name == "get_current_weather":
            result = get_current_weather(function_args["location"])

            # Add function result to messages
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(result),
                }
            )

    # Get final response with function results
    final_response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-0528",
        messages=messages,
    )

    print(final_response.choices[0].message.content)
else:
    print(response_message.content)
