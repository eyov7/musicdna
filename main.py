import gradio as gr
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def echo(audio):
    """Simple echo function to test if the basic functionality works"""
    logger.info("Echo function called")
    logger.info(f"Received audio: {audio}")
    return audio

# Create the most basic interface possible
demo = gr.Interface(
    fn=echo,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Audio(),
    title="Basic Audio Test",
    description="This is a basic test to ensure the app works on Lightning AI"
)

# For Lightning AI
app = demo.app

if __name__ == "__main__":
    logger.info("Starting application")
    demo.launch()
