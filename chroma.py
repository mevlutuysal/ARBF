# chroma.py (or your script name for populating ChromaDB)
# Make sure your ChromaDB server is running

from integration.vector_db_manager import VectorDBManager
# import uuid # Not strictly needed here as VectorDBManager can generate IDs, but good for explicit control.

# Initialize your VectorDBManager
# It will use the default collection name "aigc_governance_policies"
try:
    # The __init__ in your VectorDBManager handles connecting and getting/creating the collection.
    vector_db = VectorDBManager(host="localhost", port=8000)
    if not vector_db.is_connected():
        print("Failed to connect to ChromaDB. Aborting.")
        exit()
    # The "Using ChromaDB collection: 'aigc_governance_policies'" message will come from your VectorDBManager's __init__
except Exception as e:
    print(f"Error initializing VectorDBManager: {e}")
    exit()

print(vector_db.collection.get()['ids'])
vector_db.collection.delete(ids=[
    "policy_cc_by",
    "policy_cc_by_sa",
    "policy_private"
])
print(vector_db.collection.get()['ids'])
# --- Define your policy documents and metadata for all 7 scenarios ---
# Using the placeholder URIs as discussed since you don't need a local server

policies_to_add = [
    {
        "id": "platform-standard-commercial-v1", # SCENARIO-001
        "text": """This Standard Metaverse Commercial License (Version 1) grants the buyer the right to use the purchased AI-Generated Content (AIGC) for commercial purposes within approved Metaverse platforms. The creator retains all underlying intellectual property rights to the original design. The buyer is purchasing a license to use an instance of the AIGC. Resale of this specific licensed instance is permitted under the original sale terms. Modification of the AIGC is not permitted without explicit written permission from the original creator. This license is non-exclusive unless otherwise specified.""",
        "metadata": {
            "url": "https://metaverse.yourplatform.com/licenses/standard-commercial-v1",
            "license_type": "Platform Standard Commercial",
            "description": "Permits commercial use of AIGC instance in Metaverse, creator retains IP, resale of instance allowed, no modification without permission."
        }
    },
    {
        "id": "cc-by-4.0", # SCENARIO-002
        "text": """You are free to:
Share — copy and redistribute the material in any medium or format for any purpose, even commercially.
Adapt — remix, transform, and build upon the material for any purpose, even commercially.
The licensor cannot revoke these freedoms as long as you follow the license terms.
Under the following terms:
Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.""",
        "metadata": {
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "license_type": "CC BY 4.0",
            "description": "Creative Commons Attribution 4.0 International. Allows sharing and adaptation for any purpose, even commercially, with attribution."
        }
    },
    {
        "id": "platform-exclusive-commercial-royalty-v1", # SCENARIO-003
        "text": """This Exclusive Metaverse Commercial License with Royalty (Version 1) grants the licensee sole and exclusive rights to use the specified AI-Generated Content (AIGC) for commercial purposes within the agreed-upon Metaverse platforms or projects. The original creator retains copyright but licenses out all commercial exploitation rights to the licensee for the duration of this agreement. The licensee agrees to pay the creator a royalty of 5% (example percentage) of net revenue generated from the use of the AIGC. No other party, including the creator, may commercially exploit this AIGC during the term of this license. Modifications by the licensee are permitted. This license is transferable only with the express written consent of the original creator.""",
        "metadata": {
            "url": "https://metaverse.yourplatform.com/licenses/exclusive-commercial-royalty-v1",
            "license_type": "Platform Exclusive Commercial with Royalty",
            "description": "Grants exclusive commercial rights for the AIGC to one entity, with ongoing royalty payments to the creator."
        }
    },
    {
        "id": "cc0-1.0", # SCENARIO-004
        "text": """The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law. You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.""",
        "metadata": {
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "license_type": "CC0 1.0 Universal",
            "description": "Public Domain Dedication. Allows copying, modification, distribution, and performance, even for commercial purposes, without asking permission."
        }
    },
    {
        "id": "platform-nonexclusive-royaltyfree-music-v1", # SCENARIO-005
        "text": """This Non-Exclusive Royalty-Free Metaverse Music License (Version 1) grants the buyer a non-exclusive, worldwide, perpetual right to use the purchased audio track as background or ambient music within their owned or controlled Metaverse experiences, games, or virtual spaces. This license is royalty-free, meaning no further payments are due to the creator after the initial purchase for these permitted uses. The buyer may not resell, re-license, or distribute the audio track as a standalone product or as part of another stock media offering. The creator retains full copyright and ownership of the audio track.""",
        "metadata": {
            "url": "https://metaverse.yourplatform.com/licenses/nonexclusive-royaltyfree-music-v1",
            "license_type": "Platform Non-Exclusive Royalty-Free Music",
            "description": "Allows use of music in Metaverse projects for a one-time fee, non-exclusive. Creator retains copyright."
        }
    },
    {
        "id": "platform-personal-use-only-v1", # SCENARIO-006
        "text": """This Personal Use Only License (Version 1) for AI-Generated Content (AIGC) grants the original creator the right to use the AIGC for their own personal, non-commercial purposes within the Metaverse platform. The AIGC may not be sold, licensed, publicly distributed, or used for any commercial activity. Ownership of this specific instance is recorded for the creator, but no broader usage rights are conferred or implied for others.""",
        "metadata": {
            "url": "https://metaverse.yourplatform.com/licenses/personal-use-only-v1",
            "license_type": "Platform Personal Use Only",
            "description": "Restricts AIGC use to the creator's personal, non-commercial activities on the platform."
        }
    },
    {
        "id": "platform-unique-collectible-commercial-owner-v1", # SCENARIO-007
        "text": """This Unique Digital Collectible License (Version 1) signifies that the buyer owns the specific Non-Fungible Token (NFT) representing this AI-Generated Content (AIGC) and is granted the right to display and use this unique instance for commercial purposes (e.g., as an avatar in monetized streams, in promotional material for their own Metaverse ventures). The original creator retains the underlying intellectual property of the design concept and may create other variations, but this specific tokenized version is unique. The buyer may resell this specific NFT, and the accompanying commercial use rights transfer to the new owner. Significant modifications to the visual appearance of the AIGC by the owner are not permitted without creator consent.""",
        "metadata": {
            "url": "https://metaverse.yourplatform.com/licenses/unique-collectible-commercial-owner-v1",
            "license_type": "Platform Unique Collectible - Commercial Use by Owner",
            "description": "Buyer owns the unique NFT and can use it commercially; resale of NFT transfers these rights."
        }
    }
]

# --- Add to ChromaDB using the correct method from VectorDBManager ---
# print(f"\nAttempting to add {len(policies_to_add)} policies to ChromaDB using 'add_policy_document'...")
# added_count = 0
# failed_ids = []
#
# for policy in policies_to_add:
#     try:
#         # Using the identified method from your VectorDBManager
#         vector_db.add_policy_document(
#             text=policy["text"],
#             doc_id=policy["id"], # Pass the explicit ID
#             metadata=policy["metadata"]
#         )
#         # The success message "Added document '{doc_id}'..." will now come from your VectorDBManager method
#         added_count += 1
#     except Exception as e:
#         # The error message "Error adding document '{doc_id}'..." will come from your VectorDBManager
#         # or this generic catch if the method call itself fails for other reasons.
#         print(f"Outer error trying to add policy with id {policy['id']}: {e}")
#         failed_ids.append(policy['id'])
#
# print(f"\nFinished adding policies. {added_count} policies reported as processed by VectorDBManager.")
# if failed_ids:
#     print(f"Policies that might have encountered issues (check logs from VectorDBManager): {', '.join(failed_ids)}")