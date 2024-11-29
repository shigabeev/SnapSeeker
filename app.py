import gradio as gr
import os
import boto3
import requests
import shutil
from dotenv import load_dotenv
from pexels_api import API
from DPF.pipelines import FilterPipeline
from DPF import ShardsDatasetConfig, DatasetReader, FilesDatasetConfig
from DPF.filters.images.face_focus_filter import FaceFocusFilter
from DPF.filters.images.grayscale_filter import GrayscaleFilter
from DPF.filters.images.noise_estimation_filter import NoiseEstimationFilter
import asyncio
import aiohttp
import aiofiles
from PIL import Image, UnidentifiedImageError
import io
import pandas as pd
import tempfile
from urllib.parse import urlparse
import random
import json
import sys
from joblib import Parallel, delayed
from urllib.parse import urlparse
import os
import logging
from contextlib import redirect_stdout
import hashlib
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize Pexels API client
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
pexels_api = API(PEXELS_API_KEY)

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_API_HOST = os.getenv("GOOGLE_SEARCH_API_HOST")
GOOGLE_SEARCH_BASE_URL = os.getenv("GOOGLE_SEARCH_BASE_URL")    

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

FILTERS = [
    ("face_focus", FaceFocusFilter(threshold=2000.0, detect_face=True, workers=1, batch_size=1)),
    ("focus", FaceFocusFilter(threshold=2000.0, detect_face=False, workers=1, batch_size=1)),
    ("grayscale", GrayscaleFilter(workers=1, batch_size=1)),
    ("noise_filter", NoiseEstimationFilter(model_path='/home/ubuntu/filter images/noise_estimator_model.joblib',
                                    params_path='/home/ubuntu/filter images/feature_params.joblib',
                                    workers=1,
                                    batch_size=1))
]

def create_csv_filename(folder_path, base_name='images.csv'):
    counter = 1
    file_name = base_name
    while os.path.exists(os.path.join(folder_path, file_name)):
        file_name = f"{os.path.splitext(base_name)[0]}_{counter}.csv"
        counter += 1
    return os.path.join(folder_path, file_name)

def create_temp_csv(images):
    folder_path = 'csv_files'
    os.makedirs(folder_path, exist_ok=True)
    csv_file_path = create_csv_filename(folder_path)
    df = pd.DataFrame(images, columns=['query', 'image_path'])
    df.to_csv(csv_file_path, index=False)
    return csv_file_path

def verify_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, SyntaxError) as e:
        print(f"Image verification failed for {image_path}: {e}")
        return False

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except (IOError, SyntaxError) as e:
        print(f"Failed to get image size for {image_path}: {e}")
        return None

def apply_filters(csv_path, filters):
    config = FilesDatasetConfig.from_path_and_columns(csv_path, image_path_col='image_path')
    reader = DatasetReader()
    processor = reader.read_from_config(config)

    for filter_name, filter_obj in filters:
        processor.apply_data_filter(filter_obj)
        filtered_csv = f'filtered_{filter_name}.csv'
        processor.df.to_csv(filtered_csv, index=False)
        
        # Create a new processor with the filtered results
        config = FilesDatasetConfig.from_path_and_columns(filtered_csv, image_path_col='image_path')
        processor = reader.read_from_config(config)

    # Return only the rows that passed all filters
    return processor.df[processor.df.apply(lambda row: all(row[f'{filter_name}_pass'] for filter_name, _ in filters), axis=1)]

def parse_pages(pages):
    if '-' in pages:
        pages_start, pages_end = pages.split('-', maxsplit=1)
        pages_start, pages_end = int(pages_start.strip()), int(pages_end.strip())
    else:
        pages_start = 0
        pages_end = int(pages.strip())

    return pages_start, pages_end

def prepare_search_queries(search_queries, pages_start, pages_end):
    return [(query, page) for page in range(pages_start, pages_end) for query in search_queries]

def create_directories(save_path):
    directories = {
        "unprocessed": os.path.abspath(os.path.join(save_path, "unprocessed")),
        "accepted": os.path.abspath(os.path.join(save_path, "accepted")),
        "rejected": os.path.abspath(os.path.join(save_path, "rejected"))
    }
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    return directories

def apply_filters(csv_path, filters):
    config = FilesDatasetConfig.from_path_and_columns(csv_path, image_path_col='image_path')
    reader = DatasetReader()
    processor = reader.read_from_config(config)

    for filter_name, filter_obj in filters:
        processor.apply_data_filter(filter_obj)
        filtered_csv = f'filtered_{filter_name}.csv'
        processor.df.to_csv(filtered_csv, index=False)
        
        # Create a new processor with the filtered results
        config = FilesDatasetConfig.from_path_and_columns(filtered_csv, image_path_col='image_path')
        processor = reader.read_from_config(config)

    # Return only the rows that passed all filters
    df = processor.df
    df.columns = df.columns.str.replace('_x', '').str.replace('_y', '')
    df = df.loc[:, ~df.columns.duplicated()]
    return processor.df[processor.df.apply(lambda row: all(row[f'{filter_name}_pass'] for filter_name, _ in filters), axis=1)]

class ImageDownloader:

    def __init__(self, source, directories, min_resolution=1024):
        self.source = source
        self.session = requests.session()
        self.directories = directories
        self.min_resolution = min_resolution
        self.seen_hashes = self._load_existing_hashes()
        
    def _load_existing_hashes(self):
        """Load precomputed hashes from all existing dataset folders"""
        print("Loading existing hashes...")
        seen_hashes = set()
        
        # Find all dataset folders that have accepted/rejected subdirs
        root_dir = Path('.')
        dataset_dirs = [d for d in root_dir.iterdir() if d.is_dir() and 
                       (d / "accepted").exists() and (d / "rejected").exists()]
        
        for dataset_dir in dataset_dirs:
            hash_file = dataset_dir / "image_hashes.json"
            if hash_file.exists():
                try:
                    with open(hash_file) as f:
                        hashes = json.load(f)
                        seen_hashes.update(hashes)
                    logging.info(f"Loaded {len(hashes)} cached hashes from {dataset_dir}")
                except Exception as e:
                    logging.error(f"Error loading hashes from {dataset_dir}: {e}")
            else:
                # Calculate hashes for existing images
                hashes = self._calculate_directory_hashes(dataset_dir)
                try:
                    with open(hash_file, 'w') as f:
                        json.dump(list(hashes), f)
                    seen_hashes.update(hashes)
                    logging.info(f"Calculated and cached {len(hashes)} hashes for {dataset_dir}")
                except Exception as e:
                    logging.error(f"Error saving hashes for {dataset_dir}: {e}")
                    
        return seen_hashes

    def _calculate_directory_hashes(self, directory):
        """Calculate MD5 hashes for all images in accepted and rejected folders"""
        hashes = set()
        for subdir in ["accepted", "rejected"]:
            dir_path = directory / subdir
            if not dir_path.exists():
                continue
                
            for img_path in dir_path.rglob("*"):
                if not img_path.is_file():
                    continue
                try:
                    with open(img_path, 'rb') as f:
                        content_hash = hashlib.md5(f.read()).hexdigest()
                        hashes.add(content_hash)
                except Exception as e:
                    logging.error(f"Error calculating hash for {img_path}: {e}")
        
        return hashes

    def _save_new_hash(self, content_hash, save_path):
        """Save new hash to the dataset's hash file"""
        hash_file = Path(save_path).parent / "image_hashes.json"
        try:
            existing_hashes = []
            if hash_file.exists():
                with open(hash_file) as f:
                    existing_hashes = json.load(f)
            
            existing_hashes.append(content_hash)
            
            with open(hash_file, 'w') as f:
                json.dump(existing_hashes, f)
        except Exception as e:
            logging.error(f"Error saving hash to {hash_file}: {e}")

    def search_by_query(self, search_query, page):
        if self.source == "pexels":
            pexels_api.search(search_query, page=page, results_per_page=80)
            photos = pexels_api.get_entries()
            photos = [photo.original for photo in photos]
            return photos
        elif self.source == "Google search":
            print(f"Searching for {search_query} on page {page}")
            headers = {
                'x-rapidapi-key': GOOGLE_SEARCH_API_KEY,
                'x-rapidapi-host': GOOGLE_SEARCH_API_HOST
            }
            
            params = {
                'query': search_query,
                'size': '2mp_and_more',
                'type': 'photo',
                'safe_search': 'on',
                'region': 'us',
                # "limit":10,
            }
            
            photos = []
            params['page'] = page
            response = requests.get(GOOGLE_SEARCH_BASE_URL, headers=headers, params=params)
            data = response.json()
            photos = [img['thumbnail_url'] for img in data['data']]
            return photos
        else:
            raise ValueError(f"Unsupported source type: {self.source}")

    def _get_image_path(self, url, query):
        path = urlparse(url).path
        ext = os.path.splitext(path)[-1].lstrip('.').replace('/', '_')
        basename = f"{query}_{os.urandom(8).hex()}.{ext}"
        return os.path.join(self.directories["unprocessed"], query, basename)

    def _get_rejected_path(self, image_path):
        basename = os.path.basename(image_path)
        return os.path.join(self.directories["rejected"], basename)
    
    def download_one_image(self, url, query):
        try:
            response = self.session.get(url, timeout=(2, 20))
        except Exception as e:
            logging.error(f"Error processing url {url} for query '{query}': {e}")
            return None
        if response.status_code != 200:
            return None

        # Calculate MD5 of image content before saving
        content_hash = hashlib.md5(response.content).hexdigest()
        
        # Check if we've seen this hash before
        if content_hash in self.seen_hashes:
            logging.info(f"Duplicate image found for {url}, skipping...")
            return None
        
        # Save new hash
        self.seen_hashes.add(content_hash)
        
        # Save the unique image
        image_path = self._get_image_path(url, query)
        with open(image_path, 'wb') as img_file:
            img_file.write(response.content)
            
        # Save the hash to the dataset's hash file
        self._save_new_hash(content_hash, image_path)

        # Perform remaining checks
        try:
            img = Image.open(image_path)
            img.verify()
            width, height = img.size
            if width < self.min_resolution or height < self.min_resolution:
                new_fp = self._get_rejected_path(image_path)
                shutil.move(image_path, new_fp)
                return None
        except Exception as e:
            logging.error(f"Error processing url {url} for query '{query}': {e}")
            new_fp = self._get_rejected_path(image_path)
            shutil.move(image_path, new_fp)
            return None

        # Update extension based on actual format
        img_format = img.format.lower()
        path, ext = os.path.splitext(image_path)
        new_fp = f"{path}.{img_format}"
        shutil.move(image_path, new_fp)
        return new_fp

    def download_batch(self, urls, folder_name):
        os.makedirs(os.path.join(self.directories["unprocessed"], folder_name), exist_ok=True)
        image_paths = Parallel(n_jobs=1)(
            delayed(self.download_one_image)(url, folder_name) 
            for url in urls
        )
        return [path for path in image_paths if path is not None]

def create_dataset(state, source, search_queries, filters, min_resolution, save_path, pages, progress = gr.Progress()):
    print(f"Creating dataset with source: {source}, search queries: {search_queries}, filters: {filters}, min resolution: {min_resolution}, save path: {save_path}, pages: {pages}")
    # Parse the minimum resolution
    min_resolution = int(min_resolution)

    directories = create_directories(save_path)

    pages_start, pages_end = parse_pages(pages)
    print(f"Pages start: {pages_start}, pages end: {pages_end}")

    filters = [f for f in FILTERS if f[0] in filters]

    downloader = ImageDownloader(source=source, 
                                directories=directories,
                                min_resolution=min_resolution)
    print(f"Downloader initialized with directories: {directories}")

    folder_path = 'csv_files'
    os.makedirs(folder_path, exist_ok=True)

    print("Search queries: ", prepare_search_queries(search_queries.splitlines(), pages_start, pages_end))


    for query, page in prepare_search_queries(search_queries.splitlines(), pages_start, pages_end): # progress.tqdm(
        print(f"Downloading images for query: {query}")
        urls = downloader.search_by_query(query, page)
        image_paths = downloader.download_batch(urls, query)

        csv_file_path = create_csv_filename(folder_path)
        df = pd.DataFrame(image_paths, columns=['image_path'])
        df.to_csv(csv_file_path, index=False)
        
        filtered_df = apply_filters(csv_file_path, filters)
        
        accepted_images = set(filtered_df['image_path'].to_list())
        
        os.makedirs(os.path.join(directories["accepted"], query), exist_ok=True)
        os.makedirs(os.path.join(directories["rejected"], query), exist_ok=True)

        for image_path in image_paths:
            destination_dir = directories["accepted"] if image_path in accepted_images else directories["rejected"]
            destination_dir = os.path.join(destination_dir, query)
            shutil.move(image_path, os.path.join(destination_dir, os.path.basename(image_path)))
    # state['status'] = "Dataset generation completed."
    # return state


with gr.Blocks() as demo:
    gr.Markdown("# Dataset Creator Tool")
    
    state = gr.State({'status': ''})
    
    with gr.Row():
        source = gr.Dropdown(["Google search", "Pexels"], label="Source", value="Google search")
        save_path = gr.Textbox(label="Save Path (local path or s3:// URL)")
    
    search_queries = gr.Textbox(label="Search Queries (one per line)", lines=5)
    
    with gr.Row():
        filters = gr.CheckboxGroup([x[0] for x in FILTERS], label="Filters")
        min_resolution = gr.Textbox(label="Minimum resolution (shortest side)", value="1024")
        pages = gr.Textbox(label="Pages to collect e.g. 0-3", value="0-1")
    
    with gr.Row():
        create_button = gr.Button("Create Dataset")
    
    output = gr.Textbox(label="Output", value="")
    
    create_button.click(
        fn=create_dataset,
        inputs=[state, source, search_queries, filters, min_resolution, save_path, pages],
        outputs=[output],
        # show_progress=True, 
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7865, inbrowser=False)