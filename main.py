## pip install google-genai==0.3.0

import asyncio
import json
import os
import websockets
from google import genai
import base64

# Load API key from the environment
os.environ['GOOGLE_API_KEY'] = ''
YGGDRASIL_MODEL = "gemini-2.0-flash-exp"  # Use your model ID

valkyrie_client = genai.Client(
    http_options={
        'api_version': 'v1alpha',
    }
)

# Mock function for lighting adjustments (representing Mjölnir's power)
def wield_mjolnir(luminosity, aura_hue):
    """Adjusts light settings using Mjölnir's strength."""
    return {
        "luminosity": luminosity,
        "auraHue": aura_hue,
    }

# Define the divine tool
tool_wield_mjolnir = {
    "function_declarations": [
        {
            "name": "wield_mjolnir",
            "description": "Channel the power of Mjölnir to adjust light luminosity and aura hue.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "luminosity": {
                        "type": "NUMBER",
                        "description": "Light intensity from 0 to 100. Zero signifies darkness, and 100 represents full illumination."
                    },
                    "aura_hue": {
                        "type": "STRING",
                        "description": "The hue of light aura, which can be `daylight`, `cool`, or `warm`."
                    }
                },
                "required": ["luminosity", "aura_hue"]
            }
        }
    ]
}

async def bifrost_handler(heimdall_connection: websockets.WebSocketServerProtocol):
    """Handles the interaction with Yggdrasil API via Bifrost (WebSocket).

    Args:
        heimdall_connection: The WebSocket connection representing Heimdall's watch.
    """
    try:
        asgardian_message = await heimdall_connection.recv()
        asgardian_data = json.loads(asgardian_message)
        asgardian_config = asgardian_data.get("setup", {})
        
        asgardian_config["tools"] = [tool_wield_mjolnir]
        
        async with valkyrie_client.aio.live.connect(model=YGGDRASIL_MODEL, config=asgardian_config) as yggdrasil_session:
            print("Connected to Yggdrasil API")

            async def odin_to_yggdrasil():
                """Transfers messages from Heimdall to Yggdrasil."""
                try:
                    async for heimdall_message in heimdall_connection:
                        try:
                            bifrost_data = json.loads(heimdall_message)
                            if "realtime_input" in bifrost_data:
                                for fragment in bifrost_data["realtime_input"]["media_chunks"]:
                                    if fragment["mime_type"] == "audio/pcm":
                                        await yggdrasil_session.send({"mime_type": "audio/pcm", "data": fragment["data"]})
                                        
                                    elif fragment["mime_type"] == "image/jpeg":
                                        await yggdrasil_session.send({"mime_type": "image/jpeg", "data": fragment["data"]})
                                        
                        except Exception as e:
                            print(f"Error sending to Yggdrasil: {e}")
                    print("Heimdall connection closed (send)")
                except Exception as e:
                    print(f"Error sending to Yggdrasil: {e}")
                finally:
                    print("odin_to_yggdrasil closed")

            async def yggdrasil_to_odin():
                """Receives divine messages from Yggdrasil and relays them to Heimdall."""
                try:
                    while True:
                        try:
                            async for yggdrasil_response in yggdrasil_session.receive():
                                if yggdrasil_response.server_content is None:
                                    if yggdrasil_response.tool_call is not None:
                                        print(f"Tool call received: {yggdrasil_response.tool_call}")
                                        tool_invocations = yggdrasil_response.tool_call.function_calls
                                        mjolnir_responses = []

                                        for invocation in tool_invocations:
                                            mjolnir_name = invocation.name
                                            mjolnir_args = invocation.args
                                            invocation_id = invocation.id

                                            if mjolnir_name == "wield_mjolnir":
                                                try:
                                                    outcome = wield_mjolnir(int(mjolnir_args["luminosity"]), mjolnir_args["aura_hue"])
                                                    mjolnir_responses.append(
                                                        {
                                                            "name": mjolnir_name,
                                                            "response": {"result": outcome},
                                                            "id": invocation_id
                                                        }
                                                    )
                                                    await heimdall_connection.send(json.dumps({"text": json.dumps(mjolnir_responses)}))
                                                    print("Mjolnir wielded successfully")
                                                except Exception as e:
                                                    print(f"Error wielding Mjolnir: {e}")
                                                    continue

                                        await yggdrasil_session.send(mjolnir_responses)
                                        continue

                                divine_turn = yggdrasil_response.server_content.model_turn
                                if divine_turn:
                                    for fragment in divine_turn.parts:
                                        if hasattr(fragment, 'text') and fragment.text is not None:
                                            await heimdall_connection.send(json.dumps({"text": fragment.text}))
                                        elif hasattr(fragment, 'inline_data') and fragment.inline_data is not None:
                                            base64_audio = base64.b64encode(fragment.inline_data.data).decode('utf-8')
                                            await heimdall_connection.send(json.dumps({
                                                "audio": base64_audio,
                                            }))
                                            print("Audio received")

                                if yggdrasil_response.server_content.turn_complete:
                                    print('\n<Turn complete>')
                        except websockets.exceptions.ConnectionClosedOK:
                            print("Heimdall connection closed normally (receive)")
                            break
                        except Exception as e:
                            print(f"Error receiving from Yggdrasil: {e}")
                            break

                except Exception as e:
                    print(f"Error receiving from Yggdrasil: {e}")
                finally:
                    print("Yggdrasil connection closed (receive)")

            odin_task = asyncio.create_task(odin_to_yggdrasil())
            thor_task = asyncio.create_task(yggdrasil_to_odin())
            await asyncio.gather(odin_task, thor_task)

    except Exception as e:
        print(f"Error in Bifrost handler: {e}")
    finally:
        print("Bifrost session closed.")

async def main() -> None:
    async with websockets.serve(bifrost_handler, "localhost", 6106):
        print("BIFROST IS READY TO CONNECT ASGARD AND MIDGARD")
        await asyncio.Future()  # Keep the server running indefinitely

if __name__ == "__main__":
    asyncio.run(main())
