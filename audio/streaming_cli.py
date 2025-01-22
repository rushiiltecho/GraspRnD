import asyncio
import os

from pynput import keyboard
# from GraspRnD.audio.openai_api_key import OPENAI_API_KEY
# from audio.openai_api_key import OPENAI_API_KEY
from voice_clients import OpenAIInteractiveVoiceClient, TurnDetectionMode
from handlers import AudioHandler, KeyboardInputHandler
from llama_index.core.tools import FunctionTool
import logging


# Add your own tools here!
# NOTE: FunctionTool parses the docstring to get description, the tool name is the function name

def detect_objects(object_name: str) -> str:
    """Detects object """
    logging.info(f"Detecting the object: {object_name}")
    # Simulated object detection logic here
    return f"detecting {object_name}"


def pick_up_object(object_name: str) -> str:
    """Picks up object"""
    logging.info(f"Picking up the object: {object_name}")
    # Simulated object pickup logic here
    return f"picking up {object_name}"


def learn_pickup_method(object_name: str) -> str:
    """Learns from humans"""
    logging.info(f"Learning pickup of object: {object_name}")
    f"learning how to pick up {object_name}"


tools = [FunctionTool.from_defaults(fn=detect_objects), FunctionTool.from_defaults(fn=pick_up_object),
         FunctionTool.from_defaults(fn=learn_pickup_method)]


async def main():
    audio_handler = AudioHandler()
    keyboard_input_handler = KeyboardInputHandler()
    keyboard_input_handler.loop = asyncio.get_running_loop()

    client = OpenAIInteractiveVoiceClient(
        api_key=os.environ.get("OPENAI_API_KEY"),
        on_text_delta=lambda text: print(f"\nAssistant: {text}", end="", flush=True),
        on_audio_delta=lambda audio: audio_handler.play_audio(audio),
        on_interrupt=lambda: audio_handler.stop_playback_immediately(),
        turn_detection_mode=TurnDetectionMode.SERVER_VAD,
        tools=tools,
    )

    # Start keyboard listener in a separate thread
    listener = keyboard.Listener(on_press=keyboard_input_handler.on_press)
    listener.start()

    try:
        await client.connect()
        message_handler = asyncio.create_task(client.handle_messages())

        print("Connected to OpenAI Realtime API!")
        print("Audio streaming will start automatically.")
        print("Press 'q' to quit")
        print("")

        # Start continuous audio streaming
        streaming_task = asyncio.create_task(audio_handler.start_streaming(client))

        # Simple input loop for quit command
        while True:
            command, _ = await keyboard_input_handler.command_queue.get()

            if command == 'q':
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        audio_handler.stop_streaming()
        audio_handler.cleanup()
        await client.close()


if __name__ == "__main__":
    print("Starting Realtime API CLI with Server VAD...")
    asyncio.run(main())