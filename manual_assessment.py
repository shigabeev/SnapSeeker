import gradio as gr
import os
import shutil
import random
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
        self.accepted_count = len(list(self.accepted_dir.rglob("*.*")))
        self.rejected_count = len(list(self.rejected_dir.rglob("*.*")))
        
        # Initialize pending images list
        self.pending_images = []
        self._refresh_pending_images()
        
    def _refresh_pending_images(self) -> None:
        """Refresh the list of pending images to be processed."""
        self.pending_images = [
            file for file in self.filtered_dir.rglob("*")
            if file.suffix.lower() in self.image_extensions
        ]
        random.shuffle(self.pending_images)
    
    def get_metadata(self, file: Path) -> dict:
            from PIL import Image
            with Image.open(file) as img:
                width, height = img.size
                return {
                    'subfolder': str(file.parent.relative_to(self.filtered_dir)),
                    'resolution': f"{width}x{height}",
                    'format': file.suffix.lower()[1:],
                }
    
    def get_next_image(self) -> Optional[Tuple[str, dict]]:
        """Get the path to the next image to be processed and metadata."""
        self.current_image = None
        
        # Try to get next image from existing list
        while self.pending_images:
            file = self.pending_images.pop()
            if file.exists():  # Double check file still exists
                self.current_image = file
                return str(file), self.get_metadata(file)
                
        # If no valid images left, refresh list and try once more
        self._refresh_pending_images()
        if self.pending_images:
            file = self.pending_images.pop()
            self.current_image = file
            return str(file), self.get_metadata(file)
        # If no images found at all, return placeholder
        self.current_image = Path("end.jpg")
        return None, {"subfolder": None, "resolution": None, "format": None}
    
    def _format_status(self, decision: Optional[bool], remaining_count: int) -> str:
        """Format status message with current counts."""
        if decision is None:
            return "All images have been processed!\n"
        return f"Image moved to {'accepted' if decision else 'rejected'}\n"

    def _handle_no_images(self) -> Tuple[str, str, None, None, None]:
        """Handle case when no images are available."""
        return "end.jpg", "No images left to process!", None, None, None

    def _move_file(self, file: Path, target_dir: Path) -> None:
        """Move file to target directory maintaining subfolder structure."""
        rel_path = file.relative_to(self.filtered_dir)
        target_path = target_dir / rel_path.parent
        target_path.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(target_path / file.name))

    def process_decision(self, decision: bool) -> Tuple[Optional[str], str, Optional[str], Optional[str], Optional[str]]:
        """Process user's decision and move the image accordingly."""
        if not self.current_image or not self.current_image.exists():
            return self._handle_no_images()
            
        try:
            # Update counters
            if decision:
                self.accepted_count += 1
            else:
                self.rejected_count += 1
                
            # Move the file
            target_dir = self.accepted_dir if decision else self.rejected_dir
            self._move_file(self.current_image, target_dir)
            
            # Get next image and prepare response
            next_image, metadata = self.get_next_image()
            remaining_count = sum(1 for _ in self.filtered_dir.rglob("*") 
                                if Path(_).suffix.lower() in self.image_extensions)
            
            if next_image is None:
                return "end.jpg", self._format_status(None, remaining_count), None, None, None
            
            status = self._format_status(decision, remaining_count)
            status += f"Stats: {self.accepted_count} accepted, {self.rejected_count} rejected, {remaining_count} remaining"
            
            return next_image, status, metadata["subfolder"], metadata["resolution"], metadata["format"]
            
        except (FileNotFoundError, OSError) as e:
            next_image, metadata = self.get_next_image()
            return next_image or "end.jpg", f"Error processing image: {str(e)}", None, None, None

def create_ui(source_dir: str) -> gr.Interface:
    
    sorter = ImageSorter(source_dir)
    first_image, metadata = sorter.get_next_image()
    
    def accept_image() -> Tuple[Optional[str], str, dict]:
        return sorter.process_decision(True)
        
    def reject_image() -> Tuple[Optional[str], str, dict]:
        return sorter.process_decision(False)
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image_display = gr.Image(value=first_image, type="filepath", label="Current Image", height="80vh", scale=1)
            with gr.Column():
                with gr.Row():
                    reject_btn = gr.Button("❌ Reject", variant="secondary")
                    accept_btn = gr.Button("✅ Accept", variant="primary")
                message = gr.Textbox(label="Status", interactive=False)
                subfolder = gr.Textbox(label="Subfolder", interactive=False, value=metadata.get("subfolder", ""))
                resolution = gr.Textbox(label="Resolution", interactive=False, value=metadata.get("resolution", ""))
                format = gr.Textbox(label="Format", interactive=False, value=metadata.get("format", ""))
        
        # Event handlers
        reject_btn.click(
            fn=reject_image,
            outputs=[image_display, message, subfolder, resolution, format]
        )
        accept_btn.click(
            fn=accept_image,
            outputs=[image_display, message, subfolder, resolution, format]
        )
    
    return demo

if __name__ == "__main__":
    # Replace with your source directory
    SOURCE_DIR = "hairstyles_5/"
    demo = create_ui(SOURCE_DIR)
    demo.launch(server_name="0.0.0.0", server_port=7866)