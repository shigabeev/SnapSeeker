import gradio as gr
from collect import create_collect_ui
from review import create_review_ui
from download import create_download_ui

def create_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Image Dataset Processing Tool")
        with gr.Tabs():
            # Automated Processing Tab
            with gr.Tab("Collect"):
                create_collect_ui()
            # Manual Assessment Tab
            with gr.Tab("Review"):
                create_review_ui()
            # Gallery Tab
            with gr.Tab("Download"):
                create_download_ui()
    
    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)