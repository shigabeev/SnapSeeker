import gradio as gr
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

class ImageSorter:
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.filtered_dir = self.source_dir / "filtered"
        self.accepted_dir = self.source_dir / "accepted"
        self.rejected_dir = self.source_dir / "rejected"
        self.current_image: Optional[Path] = None
        
        # Create directories if they don't exist
        self.accepted_dir.mkdir(exist_ok=True)
        self.rejected_dir.mkdir(exist_ok=True)
        
        # Valid image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        
        # Add counters
        self.accepted_count = 0
        self.rejected_count = 0
        self.remaining_count = sum(1 for file in self.filtered_dir.rglob("*") 
                                 if file.suffix.lower() in self.image_extensions)
        
    def get_next_image(self) -> Optional[str]:
        """Get the path to the next image to be processed."""
        for file in self.filtered_dir.rglob("*"):
            if file.suffix.lower() in self.image_extensions:
                self.current_image = file
                return str(file)
        return None
    
    def process_decision(self, decision: bool) -> Tuple[Optional[str], str]:
        """Process user's decision and move the image accordingly."""
        if not self.current_image:
            return None, "No more images to process!"
            
        # Update counters
        if decision:
            self.accepted_count += 1
        else:
            self.rejected_count += 1
        self.remaining_count -= 1
            
        target_dir = self.accepted_dir if decision else self.rejected_dir
        shutil.move(str(self.current_image), str(target_dir / self.current_image.name))
        
        next_image = self.get_next_image()
        status = f"Image moved to {'accepted' if decision else 'rejected'}\n"
        status += f"Stats: {self.accepted_count} accepted, {self.rejected_count} rejected, {self.remaining_count} remaining"
        
        return next_image, status

def create_ui(source_dir: str) -> gr.Interface:
    sorter = ImageSorter(source_dir)
    first_image = sorter.get_next_image()
    
    def accept_image() -> Tuple[Optional[str], str]:
        return sorter.process_decision(True)
        
    def reject_image() -> Tuple[Optional[str], str]:
        return sorter.process_decision(False)
    
    with gr.Blocks() as demo:
        with gr.Row():
            image_display = gr.Image(value=first_image, type="filepath", label="Current Image")
        with gr.Row():
            reject_btn = gr.Button("❌ Reject", variant="secondary")
            accept_btn = gr.Button("✅ Accept", variant="primary")
        message = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        reject_btn.click(
            fn=reject_image,
            outputs=[image_display, message]
        )
        accept_btn.click(
            fn=accept_image,
            outputs=[image_display, message]
        )
    
    return demo

if __name__ == "__main__":
    # Replace with your source directory
    SOURCE_DIR = "hairstyles_4/"
    demo = create_ui(SOURCE_DIR)
    demo.launch(server_name="0.0.0.0", server_port=7866)