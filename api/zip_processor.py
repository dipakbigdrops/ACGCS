import io
import logging
import os
import zipfile
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from api.models import ExtractedText

logger = logging.getLogger(__name__)


class ZipProcessor:
    """Memory-efficient ZIP file processor that extracts and processes files one at a time."""
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024):
        self.max_file_size = max_file_size
        self.supported_image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        self.supported_html_extensions = {'.html', '.htm'}
        self.supported_extensions = self.supported_image_extensions | self.supported_html_extensions
    
    async def extract_files_sequential(
        self, 
        zip_bytes: bytes
    ) -> AsyncGenerator[Tuple[str, str, bytes], None]:
        """
        Extract files from ZIP one at a time to minimize memory usage.
        Yields: (file_path, file_type, file_bytes)
        file_type: 'image' or 'html'
        """
        try:
            zip_buffer = io.BytesIO(zip_bytes)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                file_list = zip_file.namelist()
                logger.info(f"ZIP contains {len(file_list)} total entries")
                
                processed_count = 0
                skipped_count = 0
                
                for file_path in file_list:
                    if file_path.endswith('/') or file_path.endswith('\\'):
                        continue
                    
                    filename = os.path.basename(file_path)
                    if filename.startswith('.') or filename.startswith('__'):
                        skipped_count += 1
                        continue
                    
                    file_extension = os.path.splitext(filename)[1].lower()
                    
                    if file_extension not in self.supported_extensions:
                        skipped_count += 1
                        logger.debug(f"Skipping unsupported file: {file_path}")
                        continue
                    
                    try:
                        file_info = zip_file.getinfo(file_path)
                        
                        if file_info.file_size == 0:
                            logger.warning(f"Skipping empty file: {file_path}")
                            skipped_count += 1
                            continue
                        
                        if file_info.file_size > self.max_file_size:
                            logger.warning(
                                f"Skipping large file: {file_path} "
                                f"({file_info.file_size / (1024*1024):.1f}MB exceeds limit)"
                            )
                            skipped_count += 1
                            continue
                        
                        file_bytes = zip_file.read(file_path)
                        
                        if file_extension in self.supported_image_extensions:
                            file_type = 'image'
                        else:
                            file_type = 'html'
                        
                        processed_count += 1
                        logger.info(f"Processing file {processed_count}: {file_path} ({file_type})")
                        
                        yield (file_path, file_type, file_bytes)
                        
                        del file_bytes
                        
                    except Exception as e:
                        logger.error(f"Error extracting file {file_path}: {e}")
                        skipped_count += 1
                        continue
                
                logger.info(
                    f"ZIP processing complete: {processed_count} files processed, "
                    f"{skipped_count} files skipped"
                )
                
        except zipfile.BadZipFile:
            logger.error("Invalid ZIP file format")
            raise ValueError("Invalid ZIP file format")
        except Exception as e:
            logger.error(f"Error processing ZIP file: {e}")
            raise
    
    def validate_zip_structure(self, zip_bytes: bytes) -> Tuple[bool, str, Dict]:
        """
        Validate ZIP file structure and return statistics.
        Returns: (is_valid, error_message, stats)
        """
        try:
            zip_buffer = io.BytesIO(zip_bytes)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                file_list = zip_file.namelist()
                
                if not file_list:
                    return False, "ZIP file is empty", {}
                
                stats = {
                    'total_files': 0,
                    'image_files': 0,
                    'html_files': 0,
                    'unsupported_files': 0,
                    'total_size': 0,
                    'nested_folders': False
                }
                
                for file_path in file_list:
                    if file_path.endswith('/') or file_path.endswith('\\'):
                        continue
                    
                    if '/' in file_path or '\\' in file_path:
                        stats['nested_folders'] = True
                    
                    filename = os.path.basename(file_path)
                    
                    if filename.startswith('.') or filename.startswith('__'):
                        continue
                    
                    file_extension = os.path.splitext(filename)[1].lower()
                    
                    try:
                        file_info = zip_file.getinfo(file_path)
                        stats['total_size'] += file_info.file_size
                        
                        if file_extension in self.supported_image_extensions:
                            stats['image_files'] += 1
                            stats['total_files'] += 1
                        elif file_extension in self.supported_html_extensions:
                            stats['html_files'] += 1
                            stats['total_files'] += 1
                        else:
                            stats['unsupported_files'] += 1
                    except Exception:
                        continue
                
                if stats['total_files'] == 0:
                    return False, "No supported files found in ZIP (must contain .html, .png, .jpg, etc.)", stats
                
                return True, "", stats
                
        except zipfile.BadZipFile:
            return False, "Invalid ZIP file format", {}
        except Exception as e:
            return False, f"Error validating ZIP: {str(e)}", {}
