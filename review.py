import os
import shutil
import random
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

class ImageSorter:
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.accepted_dir = self.source_dir / "accepted"
        self.reviewed_dir = self.accepted_dir / "reviewed"
        self.rejected_dir = self.source_dir / "rejected"
        self.current_image: Optional[Path] = None
        
        # Create directories if they don't exist
        self.reviewed_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(exist_ok=True)
        (self.accepted_dir / 'other').mkdir(parents=True, exist_ok=True)
        
        # Valid image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        
        # Add counters
        self.accepted_count = len(list(self.accepted_dir.rglob("*.*")))
        self.rejected_count = len(list(self.rejected_dir.rglob("*.*")))
        
        # Initialize pending images list
        self.pending_images = []
        self._refresh_pending_images()
        
        # Add method to get all subfolders
        self.available_subfolders = self._get_all_subfolders()
        
        # Add gallery state
        self.current_gallery_folder = None
        self.gallery_images = []
    
    def _refresh_pending_images(self) -> None:
        """Refresh the list of pending images to be processed."""
        self.pending_images = [
            file for file in self.accepted_dir.rglob("*")
            if file.suffix.lower() in self.image_extensions
            and self.accepted_dir / "reviewed" not in file.parents
        ]
        random.shuffle(self.pending_images)
    
    def _get_all_subfolders(self) -> list[str]:
        """Get all unique subfolder paths relative to accepted_dir."""
        subfolders = set()
        for file in self.accepted_dir.rglob("*"):
            if file.is_file() and file.suffix.lower() in self.image_extensions:
                if self.reviewed_dir not in file.parents:
                    rel_path = str(file.parent.relative_to(self.accepted_dir))
                    if rel_path != '.':  # Skip root folder
                        subfolders.add(rel_path)
        return sorted(list(subfolders))
    
    def get_metadata(self, file: Path) -> dict:
            from PIL import Image
            with Image.open(file) as img:
                width, height = img.size
                return {
                    'subfolder': str(file.parent.relative_to(self.accepted_dir)),
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

    def _move_file(self, file: Path, target_dir: Path, new_subfolder: Optional[str] = None) -> None:
        """Move file to target directory with optional subfolder override."""
        if new_subfolder is not None:
            target_path = target_dir / new_subfolder
        else:
            rel_path = file.relative_to(self.accepted_dir)
            target_path = target_dir / rel_path.parent
        
        target_path.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file), str(target_path / file.name))

    def process_decision(self, decision: bool, new_subfolder: Optional[str] = None) -> Tuple[Optional[str], str, Optional[str], Optional[str], Optional[str]]:
        """Process user's decision and move the image accordingly."""
        if not self.current_image or not self.current_image.exists():
            return self._handle_no_images()
            
        try:
            # Update counters
            if decision:
                self.accepted_count += 1
            else:
                self.rejected_count += 1
                
            # Move the file with potentially new subfolder
            target_dir = self.accepted_dir if decision else self.rejected_dir
            self._move_file(self.current_image, target_dir, new_subfolder)
            
            # Get next image and prepare response
            next_image, metadata = self.get_next_image()
            remaining_count = sum(1 for _ in self.accepted_dir.rglob("*") 
                                if Path(_).suffix.lower() in self.image_extensions)
            
            if next_image is None:
                return "end.jpg", self._format_status(None, remaining_count), None, None, None
            
            status = self._format_status(decision, remaining_count)
            status += f"Stats: {self.accepted_count} accepted, {self.rejected_count} rejected, {remaining_count} remaining"
            
            return next_image, status, metadata["subfolder"], metadata["resolution"], metadata["format"]
            
        except (FileNotFoundError, OSError) as e:
            next_image, metadata = self.get_next_image()
            return next_image or "end.jpg", f"Error processing image: {str(e)}", None, None, None

    def load_gallery(self, subfolder: str) -> list[Tuple[str, dict]]:
        """Load all images from a specific subfolder in accepted directory."""
        if not subfolder:
            return []
            
        folder_path = self.accepted_dir / subfolder
        images = []
        
        if folder_path.exists():
            for file in folder_path.glob("*.*"):
                if file.suffix.lower() in self.image_extensions:
                    images.append((str(file), self.get_metadata(file)))
                    
        self.current_gallery_folder = subfolder
        self.gallery_images = images
        return images

def get_valid_source_dirs() -> list[str]:
    """Find all folders in 'collected_datasets' that have 'accepted' and 'rejected' subfolders."""
    base_dir = Path('collected_datasets')
    valid_dirs = []
    
    if base_dir.exists():
        for folder in base_dir.iterdir():
            if folder.is_dir():
                # Check if folder has both accepted and rejected subdirectories
                if (folder / 'accepted').exists() and (folder / 'rejected').exists():
                    valid_dirs.append(str(folder))
    
    return sorted(valid_dirs)

def create_review_ui() -> gr.Blocks:
    valid_dirs = get_valid_source_dirs()
    default_dir = valid_dirs[0] if valid_dirs else "No valid directories found"
    sorter = ImageSorter(default_dir)
    first_image, metadata = sorter.get_next_image()
    
    def switch_directory(new_dir: str) -> Tuple[str, str, str, Optional[str], Optional[str]]:
        nonlocal sorter
        sorter = ImageSorter(new_dir)
        first_image, metadata = sorter.get_next_image()
        # Create new dropdown with updated choices instead of trying to update existing one
        return (
            first_image,
            "Switched to directory: " + new_dir,
            gr.Dropdown(choices=sorter.available_subfolders, value=sorter.available_subfolders[0] if sorter.available_subfolders else None),
            metadata.get("resolution", ""),
            metadata.get("format", "")
        )
    
    def accept_image(subfolder: str) -> Tuple[Optional[str], str, str, Optional[str], Optional[str]]:
        return sorter.process_decision(True, subfolder)
        
    def reject_image(subfolder: str) -> Tuple[Optional[str], str, str, Optional[str], Optional[str]]:
        return sorter.process_decision(False, subfolder)
    
    with gr.Blocks() as demo:
        with gr.Row():
            source_dir = gr.Dropdown(
                choices=valid_dirs,
                value=default_dir,
                label="Source Directory",
                interactive=True
            )
        
        with gr.Row():
            with gr.Column():
                image_display = gr.Image(value=first_image, 
                                       type="filepath", 
                                       label="Current Image",
                                       height="80vh")
            with gr.Column():
                with gr.Row():
                    accept_btn = gr.Button("✅ Accept", variant="primary")
                    reject_btn = gr.Button("❌ Reject", variant="secondary")
                message = gr.Textbox(label="Status", interactive=False)
                subfolder = gr.Dropdown(
                    choices=sorter.available_subfolders,
                    label="Subfolder",
                    value=metadata.get("subfolder", "No subfolders found")
                )
                resolution = gr.Textbox(label="Resolution", interactive=False, value=metadata.get("resolution", ""))
                img_format = gr.Textbox(label="Format", interactive=False, value=metadata.get("format", ""))
                
        # Update event handlers to include subfolder
        reject_btn.click(
            fn=reject_image,
            inputs=[subfolder],
            outputs=[image_display, message, subfolder, resolution, img_format]
        )
        accept_btn.click(
            fn=accept_image,
            inputs=[subfolder],
            outputs=[image_display, message, subfolder, resolution, img_format]
        )
        
        # Add directory change handler
        source_dir.change(
            fn=switch_directory,
            inputs=[source_dir],
            outputs=[image_display, message, subfolder, resolution, img_format]
        )

    return demo

if __name__ == "__main__":
    demo = create_review_ui()
    demo.launch(server_name="0.0.0.0", server_port=7866)