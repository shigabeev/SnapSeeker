import os
import shutil
import random
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

class GalleryDownloader:
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
    
    
    def _format_status(self, decision: Optional[bool], remaining_count: int) -> str:
        """Format status message with current counts."""
        if decision is None:
            return "All images have been processed!\n"
        return f"Image moved to {'accepted' if decision else 'rejected'}\n"

    def _handle_no_images(self) -> Tuple[str, str, None, None, None]:
        """Handle case when no images are available."""
        return "end.jpg", "No images left to process!", None, None, None


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

def create_download_ui() -> gr.Blocks:
    valid_dirs = get_valid_source_dirs()
    default_dir = valid_dirs[0] if valid_dirs else "No valid directories found"
    sorter = GalleryDownloader(default_dir)
    
    def create_zip_archive(folder_path: Path) -> str:
        """Create a zip archive of the specified folder and return the path to the zip file."""
        zip_path = str(folder_path) + '.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(folder_path)
                    zipf.write(file_path, arcname)
        return zip_path
    
    def load_gallery_images(folder: str) -> list[str]:
        """Load images from selected folder for gallery view"""
        images = sorter.load_gallery(folder)
        if not images:
            # Return end.jpg if no images found
            return ["end.jpg"]
        return [img[0] for img in images]
    
    def switch_directory(new_dir: str):
        nonlocal sorter
        sorter.__init__(new_dir)
        folders = sorter.available_subfolders
        
        gallery_folder = gr.Dropdown(
            choices=folders,
            label="View Folder",
            value=folders[0] if folders else None,
            interactive=bool(folders)  # Disable if no folders available
        )
        return gallery_folder

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
                gallery_folder = gr.Dropdown(
                    choices=sorter.available_subfolders,
                    label="View Folder",
                    value=sorter.available_subfolders[0]
                )
                gallery = gr.Gallery(label="Folder Contents")
            
            with gr.Column():  
                # Add download buttons
                with gr.Row():
                    accepted_download = gr.DownloadButton(
                        "📥 Download Accepted",
                        variant="secondary",
                        value=lambda: create_zip_archive(sorter.accepted_dir),
                        interactive=True if sorter.accepted_dir.exists() else False
                    )
                    reviewed_download = gr.DownloadButton(
                        "📥 Download Reviewed", 
                        variant="secondary",
                        value=lambda: create_zip_archive(sorter.reviewed_dir),
                        interactive=True if sorter.reviewed_dir.exists() else False
                    )

        
        gallery_folder.change(
            fn=load_gallery_images,
            inputs=[gallery_folder],
            outputs=[gallery]
            )
        
        # Add directory change handler
        source_dir.change(
            fn=switch_directory,
            inputs=[source_dir],
            outputs=[gallery_folder]
        )

    return demo

if __name__ == "__main__":
    demo = create_download_ui()
    demo.launch(server_name="0.0.0.0", server_port=7866)