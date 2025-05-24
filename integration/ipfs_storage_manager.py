# integration/ipfs_storage_manager.py
import requests
import json
import os
from datetime import datetime

# --- Conditional import for direct execution vs. package import ---
try:
    # This works when ipfs_storage_manager is imported as part of the 'integration' package
    # (e.g., when agent_main.py runs)
    from .utils import calculate_sha256, calculate_phash
except ImportError:
    # This is a fallback for when the script is run directly (e.g., for testing)
    # It adjusts sys.path to find the 'integration' package from the project root.
    if __name__ == '__main__' and (__package__ is None or __package__ == ''):
        import sys
        # Get the directory of the current script (e.g., /path/to/AgenticRAG/integration)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory (e.g., /path/to/AgenticRAG)
        project_root_dir = os.path.dirname(current_script_dir)
        # Add project root to sys.path
        if project_root_dir not in sys.path:
            sys.path.insert(0, project_root_dir)
        # Now try importing 'utils' from the 'integration' package
        from integration.utils import calculate_sha256, calculate_phash
        print("Note: Running ipfs_storage_manager.py directly. Adjusted sys.path for imports.")
    else:
        # If it's not a direct run and still fails, re-raise the original error
        raise


class IPFSStorageManager:
    """
    Manages interactions with Pinata for uploading content and metadata.
    Requires a Pinata JWT (Access Token).
    """
    PINATA_PIN_FILE_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    PINATA_PIN_JSON_URL = "https://api.pinata.cloud/pinning/pinJSONToIPFS"

    def __init__(self, jwt_token: str):
        """
        Initializes the manager with the Pinata JWT.

        Args:
            jwt_token: The JWT (Access Token) obtained from Pinata.
        """
        if not jwt_token:
            raise ValueError("Pinata JWT (Access Token) is required.")
        self.jwt_token = jwt_token
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}"
            # Content-Type will be set dynamically for multipart/form-data or application/json
        }
        print("IPFSStorageManager initialized for Pinata.")

    def _upload_file_to_pinata(self, file_path: str, file_name: str) -> str | None:
        """
        Private helper method to upload a physical file to Pinata.

        Args:
            file_path: The absolute path to the file to upload.
            file_name: The name to give the file on Pinata.

        Returns:
            The IPFS CID (IpfsHash) string if upload is successful, None otherwise.
        """
        try:
            with open(file_path, "rb") as fp:
                files = {"file": (file_name, fp)}
                # Pinata options can be added here if needed (e.g., wrapWithDirectory)
                # pinata_options = json.dumps({"wrapWithDirectory": False})
                # data = {'pinataOptions': pinata_options}

                response = requests.post(
                    self.PINATA_PIN_FILE_URL,
                    headers=self.headers, # JWT is in headers
                    files=files,
                    # data=data, # If using pinataOptions
                    timeout=180 # Increased timeout for larger files
                )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            # Pinata returns IpfsHash, Timestamp, PinSize, etc.
            cid = result.get("IpfsHash")
            if cid:
                # print(f"Pinata file upload successful. CID: {cid}")
                return cid
            else:
                print(f"Error uploading file to Pinata: Pinata response did not contain IpfsHash. Response: {result}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error during Pinata API request (file upload): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Pinata Response Status: {e.response.status_code}")
                try:
                    print(f"Pinata Response Body: {e.response.json()}")
                except json.JSONDecodeError:
                    print(f"Pinata Response Body (not JSON): {e.response.text}")
            return None
        except FileNotFoundError:
            print(f"Error: File not found at {file_path} for Pinata upload.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding Pinata API response (file upload): {response.text if 'response' in locals() else 'No response object'}")
            return None


    def _upload_json_to_pinata(self, json_data: dict, name: str) -> str | None:
        """
        Private helper method to upload JSON data to Pinata.

        Args:
            json_data: The Python dictionary to upload as JSON.
            name: A name for this pin (used for Pinata management).

        Returns:
            The IPFS CID (IpfsHash) string if upload is successful, None otherwise.
        """
        headers = self.headers.copy()
        headers["Content-Type"] = "application/json"

        payload = {
            "pinataOptions": {
                # "cidVersion": 1 # Optional: to get CIDv1
            },
            "pinataMetadata": {
                "name": name, # Name for the pin on Pinata
                # "keyvalues": {"customKey": "customValue"} # Optional custom metadata for Pinata
            },
            "pinataContent": json_data # The actual JSON content
        }
        try:
            response = requests.post(
                self.PINATA_PIN_JSON_URL,
                headers=headers,
                json=payload, # requests library handles json.dumps for json=
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            cid = result.get("IpfsHash")
            if cid:
                # print(f"Pinata JSON upload successful. CID: {cid}")
                return cid
            else:
                print(f"Error uploading JSON to Pinata: Pinata response did not contain IpfsHash. Response: {result}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error during Pinata API request (JSON upload): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Pinata Response Status: {e.response.status_code}")
                try:
                    print(f"Pinata Response Body: {e.response.json()}")
                except json.JSONDecodeError:
                    print(f"Pinata Response Body (not JSON): {e.response.text}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding Pinata API response (JSON upload): {response.text if 'response' in locals() else 'No response object'}")
            return None


    def upload_content(self, file_path: str) -> tuple[str | None, str | None, str | None]:
        """
        Calculates hashes, uploads a content file to Pinata, and returns its CID and hashes.

        Args:
            file_path: The path to the content file to upload.

        Returns:
            A tuple containing:
            - The CID string of the uploaded content (or None on failure).
            - The SHA-256 hash string (or None on failure).
            - The pHash string (or None if not an image or on failure).
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        print(f"Processing content file for Pinata: {file_path}")
        sha256_hash = None
        phash_val = None
        cid = None
        try:
            # 1. Calculate Hashes
            sha256_hash = calculate_sha256(file_path) # Uses the conditionally imported function
            phash_val = calculate_phash(file_path)   # Uses the conditionally imported function
            print(f"  - SHA256: {sha256_hash}")
            if phash_val:
                 print(f"  - pHash: {phash_val}")

            # 2. Upload content file
            file_name = os.path.basename(file_path)
            print(f"  - Uploading '{file_name}' to Pinata...")
            cid = self._upload_file_to_pinata(file_path, file_name)

            if cid:
                print(f"  - Pinata Upload successful. Content CID: {cid}")
            else:
                print("  - Pinata Upload failed.")
            # Return hashes even if upload fails, CID will be None
            return cid, sha256_hash, phash_val

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise # Re-raise the error
        except Exception as e:
            print(f"An unexpected error occurred during content upload with Pinata: {e}")
            # Return whatever hashes were calculated, CID will be None
            return cid, sha256_hash, phash_val


    def create_metadata(self, name: str, description: str, content_cid: str, creator_address: str, license_uri: str, sha256_hash: str | None, phash: str | None) -> dict:
        """
        Creates the metadata dictionary in a standard format.

        Args:
            name: Name of the AIGC asset.
            description: Description of the asset.
            content_cid: The CID of the primary content file (from Pinata).
            creator_address: The blockchain address of the verified creator.
            license_uri: URI pointing to the license terms (e.g., Creative Commons URL).
            sha256_hash: SHA-256 hash of the content file for integrity check.
            phash: Perceptual hash of the content file (if applicable) for similarity check.

        Returns:
            A dictionary representing the metadata JSON structure.
        """
        metadata = {
            "name": name,
            "description": description,
            "image": f"ipfs://{content_cid}", # Standard way to link content CID in NFT metadata
            "external_url": f"ipfs://{content_cid}", # Optional: Link to content
            "attributes": [
                {"trait_type": "Creator", "value": creator_address},
                {"trait_type": "License", "value": license_uri},
                {"trait_type": "Content CID", "value": content_cid}, # Explicitly include content CID
                {"trait_type": "SHA256 Hash", "value": sha256_hash} if sha256_hash else None,
                {"trait_type": "Perceptual Hash", "value": phash} if phash else None,
                {"trait_type": "Timestamp", "value": datetime.utcnow().isoformat() + "Z"}
            ],
            # Add custom properties as needed, e.g., for CAR file CID if different
            # "properties": {
            #     "car_cid": car_cid_if_different
            # }
        }
        # Remove attributes with None values
        metadata["attributes"] = [attr for attr in metadata["attributes"] if attr and attr["value"] is not None]
        print(f"Created metadata structure for '{name}'")
        return metadata

    def upload_metadata_json(self, metadata_dict: dict) -> str | None:
        """
        Uploads the metadata dictionary (as JSON) to Pinata.

        Args:
            metadata_dict: The metadata dictionary to upload.

        Returns:
            The CID string of the uploaded metadata JSON, or None on failure.
        """
        try:
            metadata_pin_name = f"{metadata_dict.get('name', 'aigc_metadata').replace(' ', '_')}_meta.json"
            print(f"Uploading metadata JSON to Pinata as '{metadata_pin_name}'...")
            cid = self._upload_json_to_pinata(metadata_dict, name=metadata_pin_name)
            if cid:
                 print(f"  - Pinata Metadata upload successful. Metadata CID: {cid}")
            else:
                 print("  - Pinata Metadata upload failed.")
            return cid
        except Exception as e:
            print(f"An error occurred during Pinata metadata upload: {e}")
            return None


# Example usage (for testing this file directly)
if __name__ == '__main__':
    # This block now correctly uses the conditionally imported functions
    # or functions from .utils if imported as a package.
    from dotenv import load_dotenv
    load_dotenv() # Load .env file from project root

    pinata_jwt = os.getenv("PINATA_JWT")
    if not pinata_jwt:
        print("Error: PINATA_JWT not found in .env file. Cannot run Pinata tests.")
    else:
        manager = IPFSStorageManager(jwt_token=pinata_jwt)

        # --- Test File Setup ---
        # Create dummy file in the same directory as the script for direct run
        # or ensure path is correct if running as module
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_content_file = os.path.join(script_dir, "test_pinata_upload.txt")

        with open(test_content_file, "w") as f:
            f.write(f"This is a test file for Pinata upload. Timestamp: {datetime.utcnow()}")
        print(f"Created test file: {test_content_file}")


        # --- Test Content Upload (Text) ---
        print(f"\n--- Testing Pinata Text Content Upload ({test_content_file}) ---")
        try:
            content_cid_txt, sha256_txt, phash_txt = manager.upload_content(test_content_file)
            if content_cid_txt:
                print(f"Pinata Text Content Upload Test SUCCESS: CID={content_cid_txt}, SHA256={sha256_txt}")

                # --- Test Metadata Upload (using text content CID) ---
                print(f"\n--- Testing Pinata Metadata Upload (for {test_content_file}) ---")
                metadata_dict = manager.create_metadata(
                    name="Test Pinata Text Asset",
                    description="A test asset based on a text file uploaded via Pinata.",
                    content_cid=content_cid_txt,
                    creator_address="0x1234567890abcdef1234567890abcdef12345678",
                    license_uri="http://example.com/test-license-pinata",
                    sha256_hash=sha256_txt,
                    phash=phash_txt # Should be None
                )
                metadata_cid = manager.upload_metadata_json(metadata_dict)
                if metadata_cid:
                     print(f"Pinata Metadata Upload Test SUCCESS: CID={metadata_cid}")
                     print(f"View Metadata: https://gateway.pinata.cloud/ipfs/{metadata_cid}")
                     print(f"View Content: https://gateway.pinata.cloud/ipfs/{content_cid_txt}")
                else:
                     print("Pinata Metadata Upload Test FAILED.")
            else:
                print("Pinata Text Content Upload Test FAILED.")
        except FileNotFoundError:
             print(f"Skipping text upload test, {test_content_file} not found during test execution.")
        except Exception as e:
             print(f"Error during Pinata text upload test: {e}")


        finally:
             # Clean up test files
             if os.path.exists(test_content_file):
                 os.remove(test_content_file)
                 print(f"Cleaned up test file: {test_content_file}")

