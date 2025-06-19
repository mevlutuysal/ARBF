# integration/utils.py
import hashlib
from PIL import Image # Requires Pillow: pip install Pillow
import imagehash # Requires imagehash: pip install imagehash
import os

def calculate_sha256(file_path: str) -> str:
    """
    Calculates the SHA-256 hash of a file.

    Args:
        file_path: Path to the file.

    Returns:
        The hex digest of the SHA-256 hash.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise

def calculate_phash(file_path: str) -> str | None:
    """
    Calculates the perceptual hash (pHash) of an image file.

    Args:
        file_path: Path to the image file.

    Returns:
        The pHash string if the file is a valid image, None otherwise.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        # Check if the file is likely an image based on extension
        # More robust checks could involve magic numbers if needed
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
        if not any(file_path.lower().endswith(ext) for ext in image_extensions):
            print(f"Info: File {os.path.basename(file_path)} is not identified as an image. Skipping pHash.")
            return None

        with Image.open(file_path) as img:
            hash_val = imagehash.phash(img)
            return str(hash_val)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        # Catch potential errors from PIL/imagehash if file is not a valid image
        print(f"Warning: Could not calculate pHash for {file_path}. Error: {e}")

        return None

# Example usage (for testing this file directly)
if __name__ == '__main__':
    # Create dummy files for testing
    dummy_text_file = "dummy_file.txt"
    dummy_image_file = "dummy_image.png" # Requires a real image file or Pillow will error

    try:
        with open(dummy_text_file, "w") as f:
            f.write("This is a test file.")

        # Create a simple dummy image using Pillow if you don't have one
        try:
            img = Image.new('RGB', (60, 30), color = 'red')
            img.save(dummy_image_file)
            print(f"Created dummy image: {dummy_image_file}")
        except Exception as img_e:
             print(f"Could not create dummy image: {img_e}. Place a real PNG named {dummy_image_file} for testing.")


        print(f"--- Testing {dummy_text_file} ---")
        sha256 = calculate_sha256(dummy_text_file)
        print(f"SHA256: {sha256}")
        phash = calculate_phash(dummy_text_file)
        print(f"pHash: {phash}") # Expected: None

        if os.path.exists(dummy_image_file):
            print(f"\n--- Testing {dummy_image_file} ---")
            sha256_img = calculate_sha256(dummy_image_file)
            print(f"SHA256: {sha256_img}")
            phash_img = calculate_phash(dummy_image_file)
            print(f"pHash: {phash_img}") # Expected: A hash string
        else:
            print(f"\nSkipping image test, {dummy_image_file} not found.")


    finally:
        # Clean up dummy files
        if os.path.exists(dummy_text_file):
            os.remove(dummy_text_file)
        if os.path.exists(dummy_image_file):
            os.remove(dummy_image_file)

