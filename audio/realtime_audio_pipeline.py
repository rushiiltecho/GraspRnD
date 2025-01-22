import asyncio
import os
import sys
from pynput import keyboard
from audio.voice_clients import OpenAIInteractiveVoiceClient, TurnDetectionMode
from audio.handlers import AudioHandler, KeyboardInputHandler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioPipeline:
    def __init__(self, api_key, turn_detection_mode=TurnDetectionMode.SERVER_VAD, tools=None, on_text_delta=None,):
        if not api_key:
            raise ValueError("API key is required")
        self.audio_handler = AudioHandler()
        self.keyboard_input_handler = KeyboardInputHandler()
        if on_text_delta is None:
            on_text_delta = lambda text: print(f"\nAssistant: {text}", end="", flush=True)
        self.client = OpenAIInteractiveVoiceClient(
        api_key=api_key,
        on_text_delta=on_text_delta,
        turn_detection_mode=turn_detection_mode,
            tools=tools or [],
        on_input_transcript=lambda transcript: print(f"\nUser: {transcript}", end="", flush=True),
        on_output_transcript=lambda transcript: print(f"\nAssistant (Transcript): {transcript}", end="", flush=True),
        )
        self.listener = None
        self.running = True

    async def start_audio_pipeline(self):
        self.keyboard_input_handler.loop = asyncio.get_running_loop()

        self.listener = keyboard.Listener(on_press=self.keyboard_input_handler.on_press)
        self.listener.start()

        try:
            await self.client.connect()
            message_handler = asyncio.create_task(self.client.handle_messages())

            logging.info("Connected to OpenAI Realtime API!")
            logging.info("Audio streaming will start automatically.")
            logging.info("Press 'q' to quit")

            streaming_task = asyncio.create_task(self.audio_handler.start_streaming(self.client))

            while self.running:
                command, _ = await self.keyboard_input_handler.command_queue.get()
                if command == 'q':
                    self.running = False
                    break

        except asyncio.CancelledError:
            logging.info("Audio pipeline cancelled")
        except Exception as e:
            logging.error(f"Error in audio pipeline: {str(e)}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        logging.info("Cleaning up audio resources...")
        self.audio_handler.stop_streaming()
        self.audio_handler.cleanup()
        await self.client.close()
        if self.listener:
            self.listener.stop()

    def stop(self):
        # Set running to False
        self.running = False
        # Stop the audio handler and client
        self.audio_handler.stop_streaming()
        self.audio_handler.cleanup()
        # Close the client
        asyncio.create_task(self.client.close())
        # Stop the listener
        if self.listener:
            self.listener.stop()

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    try:
        pipeline = AudioPipeline(api_key=api_key)
        await pipeline.start_audio_pipeline()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    logging.info("Starting Realtime API CLI with Server VAD...")
    asyncio.run(main())