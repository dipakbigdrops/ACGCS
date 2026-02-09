import pytest
from api.file_validation import validate_file_content, validate_upload_extension_and_content


def test_validate_pdf():
    assert validate_file_content("pdf", b"%PDF-1.4") is True
    assert validate_file_content("pdf", b"not a pdf") is False


def test_validate_png():
    assert validate_file_content("png", b"\x89PNG\r\n\x1a\n") is True
    assert validate_file_content("png", b"xxx") is False


def test_validate_jpeg():
    assert validate_file_content("jpg", b"\xff\xd8\xff\xe0\x00\x10") is True
    assert validate_file_content("jpeg", b"\xff\xd8\xff") is True


def test_validate_html():
    assert validate_file_content("html", b"<html>") is True
    assert validate_file_content("html", b"  <!DOCTYPE") is True
    assert validate_file_content("html", b"no tag") is False


def test_validate_upload_extension_and_content():
    ok, err = validate_upload_extension_and_content("pdf", b"%PDF", {"pdf"})
    assert ok is True
    assert err == ""

    ok, err = validate_upload_extension_and_content("txt", b"hello", {"pdf"})
    assert ok is False
    assert "Unsupported" in err

    ok, err = validate_upload_extension_and_content("pdf", b"not pdf content", {"pdf"})
    assert ok is False
    assert "content" in err.lower()
