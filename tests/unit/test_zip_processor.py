import io
import pytest
import zipfile
from api.zip_processor import ZipProcessor


def create_test_zip(files_dict):
    """Helper to create a ZIP file in memory."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename, content in files_dict.items():
            zipf.writestr(filename, content)
    return zip_buffer.getvalue()


def test_validate_valid_zip():
    """Test validation of a valid ZIP file."""
    processor = ZipProcessor()
    
    files = {
        'image1.png': b'fake image content',
        'image2.jpg': b'fake image content',
        'page.html': b'<html>test</html>',
    }
    zip_bytes = create_test_zip(files)
    
    is_valid, error_msg, stats = processor.validate_zip_structure(zip_bytes)
    
    assert is_valid
    assert error_msg == ""
    assert stats['total_files'] == 3
    assert stats['image_files'] == 2
    assert stats['html_files'] == 1


def test_validate_empty_zip():
    """Test validation of an empty ZIP file."""
    processor = ZipProcessor()
    
    files = {}
    zip_bytes = create_test_zip(files)
    
    is_valid, error_msg, stats = processor.validate_zip_structure(zip_bytes)
    
    assert not is_valid
    assert "empty" in error_msg.lower()


def test_validate_no_supported_files():
    """Test validation of ZIP with only unsupported files."""
    processor = ZipProcessor()
    
    files = {
        'readme.txt': b'text file',
        'data.json': b'{}',
        'script.js': b'console.log("test")',
    }
    zip_bytes = create_test_zip(files)
    
    is_valid, error_msg, stats = processor.validate_zip_structure(zip_bytes)
    
    assert not is_valid
    assert "no supported files" in error_msg.lower()


def test_validate_nested_folders():
    """Test validation detects nested folders."""
    processor = ZipProcessor()
    
    files = {
        'folder1/image.png': b'fake image',
        'folder1/subfolder/page.html': b'<html>test</html>',
        'image.jpg': b'fake image',
    }
    zip_bytes = create_test_zip(files)
    
    is_valid, error_msg, stats = processor.validate_zip_structure(zip_bytes)
    
    assert is_valid
    assert stats['nested_folders'] is True
    assert stats['total_files'] == 3


def test_validate_skips_hidden_files():
    """Test validation skips hidden files and system folders."""
    processor = ZipProcessor()
    
    files = {
        '.DS_Store': b'mac metadata',
        '__MACOSX/._image.png': b'mac metadata',
        'image.png': b'fake image',
    }
    zip_bytes = create_test_zip(files)
    
    is_valid, error_msg, stats = processor.validate_zip_structure(zip_bytes)
    
    assert is_valid
    assert stats['total_files'] == 1


@pytest.mark.asyncio
async def test_extract_files_sequential():
    """Test sequential file extraction from ZIP."""
    processor = ZipProcessor()
    
    files = {
        'image1.png': b'fake image 1',
        'folder/image2.jpg': b'fake image 2',
        'page.html': b'<html>test</html>',
    }
    zip_bytes = create_test_zip(files)
    
    extracted = []
    async for file_path, file_type, file_bytes in processor.extract_files_sequential(zip_bytes):
        extracted.append((file_path, file_type))
    
    assert len(extracted) == 3
    
    paths = [e[0] for e in extracted]
    assert 'image1.png' in paths
    assert 'folder/image2.jpg' in paths
    assert 'page.html' in paths
    
    types = [e[1] for e in extracted]
    assert types.count('image') == 2
    assert types.count('html') == 1


@pytest.mark.asyncio
async def test_extract_skips_empty_files():
    """Test that empty files are skipped during extraction."""
    processor = ZipProcessor()
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr('image.png', b'content')
        
        info = zipfile.ZipInfo('empty.png')
        info.file_size = 0
        zipf.writestr(info, b'')
    
    zip_bytes = zip_buffer.getvalue()
    
    extracted = []
    async for file_path, file_type, file_bytes in processor.extract_files_sequential(zip_bytes):
        extracted.append(file_path)
    
    assert len(extracted) == 1
    assert 'empty.png' not in extracted


@pytest.mark.asyncio
async def test_extract_respects_max_file_size():
    """Test that files exceeding max size are skipped."""
    max_size = 100
    processor = ZipProcessor(max_file_size=max_size)
    
    files = {
        'small.png': b'x' * 50,
        'large.png': b'x' * 200,
    }
    zip_bytes = create_test_zip(files)
    
    extracted = []
    async for file_path, file_type, file_bytes in processor.extract_files_sequential(zip_bytes):
        extracted.append(file_path)
    
    assert len(extracted) == 1
    assert 'small.png' in extracted
    assert 'large.png' not in extracted


def test_validate_invalid_zip():
    """Test validation of invalid ZIP data."""
    processor = ZipProcessor()
    
    invalid_zip = b'this is not a zip file'
    
    is_valid, error_msg, stats = processor.validate_zip_structure(invalid_zip)
    
    assert not is_valid
    assert "invalid" in error_msg.lower() or "format" in error_msg.lower()
