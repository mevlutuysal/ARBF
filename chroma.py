# In chroma.py

from integration.vector_db_manager import VectorDBManager
import sys


def populate_policies(vector_db_manager, policies):
    """Populates the collection with a list of policies."""
    # This function no longer needs to get the collection, as it's passed in.
    if not vector_db_manager.collection:
        print("ERROR: Cannot populate policies because collection is not available.")
        return

    print(f"\nAttempting to add {len(policies)} new policies to ChromaDB...")
    for i, policy in enumerate(policies):
        try:
            vector_db_manager.add_policy_document(
                text=policy["text"], doc_id=policy["id"], metadata=policy["metadata"]
            )
            print(f"  ({i + 1}/{len(policies)}) Added policy: {policy['id']}")
        except Exception as e:
            print(f"Error adding policy with id {policy['id']}: {e}")
    print("\nFinished adding policies.")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- 🚀 Starting ChromaDB Setup for ARBF Evaluation 🚀 ---")

    # This is the full list of policies with descriptive text.
    all_policies = [
        {"id": "standard-commercial-v1",
         "text": "This is a standard commercial license. It grants non-exclusive rights to use the asset in commercial projects, for-profit activities, and on marketplaces. The asset can be bought, sold, and used for business purposes.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/standard-commercial-v1", "type": "Commercial"}},
        {"id": "cc-by-4.0",
         "text": "This is the Creative Commons Attribution 4.0 license. It allows others to share, use, and adapt the work, even for commercial purposes. The only requirement is to provide proper attribution (credit) to the original creator.",
         "metadata": {"url": "https://creativecommons.org/licenses/by/4.0/", "type": "Open Source"}},
        {"id": "all-rights-reserved-v1",
         "text": "This license indicates that all rights are reserved by the creator. The work is proprietary and confidential. No distribution, modification, or use is permitted without explicit permission from the rights holder.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/all-rights-reserved-v1",
                      "type": "Proprietary"}},
        {"id": "cc0-1.0",
         "text": "This is the Creative Commons Zero (CC0) license, which dedicates the work to the public domain. The creator has waived all rights, making the asset completely free for any use, commercial or non-commercial, with no attribution required.",
         "metadata": {"url": "https://creativecommons.org/publicdomain/zero/1.0/", "type": "Public Domain"}},
        {"id": "cc-by-nc-4.0",
         "text": "This is the Creative Commons Attribution-NonCommercial 4.0 license. It allows others to share and adapt the work, but strictly for non-commercial purposes. Any commercial use or for-profit activity is forbidden. Attribution to the creator is required.",
         "metadata": {"url": "https://creativecommons.org/licenses/by-nc/4.0/", "type": "Open Source"}},
        {"id": "unique-collectible-commercial-owner-v1",
         "text": "This license is for a unique, one-of-a-kind digital collectible, such as a single-instance NFT. The owner of the asset is granted full and exclusive commercial rights to use and monetize it.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/unique-collectible-commercial-owner-v1",
                      "type": "NFT Commercial"}},
        {"id": "gpl-3.0",
         "text": "This is the GNU General Public License v3.0, a strong 'copyleft' license typically for software. It allows free use, modification, and distribution, but requires that any derivative works or software that uses it must also be licensed under the same GPL terms.",
         "metadata": {"url": "https://www.gnu.org/licenses/gpl-3.0.en.html", "type": "Copyleft"}},
        {"id": "personal-use-only-v1",
         "text": "This license is strictly for personal, non-commercial use only. The asset can be used for private display but cannot be sold, distributed, or used in any project that involves monetary gain or public distribution.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/personal-use-only-v1", "type": "Personal"}},
        {"id": "exclusive-commercial-transfer-v1",
         "text": "This license represents a full and exclusive transfer of commercial rights to a single person or entity. After the transfer, no one else, including the original creator, can use the asset for commercial purposes. The licensee gets sole rights.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/exclusive-commercial-transfer-v1",
                      "type": "Exclusive Commercial"}},
        {"id": "cc-by-nc-nd-4.0",
         "text": "This is the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 license. It allows sharing the work for non-commercial purposes, but it cannot be modified or adapted in any way. It must be shared as is, with credit to the creator.",
         "metadata": {"url": "https://creativecommons.org/licenses/by-nc-nd/4.0/", "type": "Open Source"}},
        {"id": "cc-by-sa-4.0",
         "text": "This is the Creative Commons Attribution-ShareAlike 4.0 license. It allows others to use, remix, and build upon the work, even for commercial purposes, as long as they give credit and license their new creations under the identical terms.",
         "metadata": {"url": "https://creativecommons.org/licenses/by-sa/4.0/", "type": "Copyleft"}},
        {"id": "nonexclusive-royaltyfree-music-v1",
         "text": "This is a non-exclusive, royalty-free license, typically for music or sound effects. After a one-time fee, the buyer can use the asset in unlimited commercial projects forever without paying future royalties.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/nonexclusive-royaltyfree-music-v1",
                      "type": "Royalty-Free"}},
        {"id": "editorial-use-only-v1",
         "text": "This license restricts the asset's use to editorial purposes only, such as in news articles, reporting, or documentaries. It cannot be used for commercial purposes like advertising or in products for sale.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/editorial-use-only-v1", "type": "Restricted"}},
        {"id": "MIT",
         "text": "This is the MIT License, a permissive open-source software license. It allows people to do almost anything they want with the software, including using it in proprietary software, with the only major condition being that the original copyright and license notice are included.",
         "metadata": {"url": "https://opensource.org/licenses/MIT", "type": "Permissive"}},
        {"id": "virtual-land-use-restricted-v1",
         "text": "This license governs the use of a plot of virtual land within a specific platform. The owner can build on it, but is subject to restrictions and terms of service, which may prohibit certain activities like gambling or commercial operations.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/virtual-land-use-restricted-v1",
                      "type": "Restricted Use"}},
        {"id": "platform-exclusive-asset-v1",
         "text": "This license restricts the use of an asset to a single, specific platform, game, or virtual environment. It cannot be exported or used in any other context.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/platform-exclusive-asset-v1",
                      "type": "Platform Exclusive"}},
        {"id": "cc-by-nc-sa-4.0",
         "text": "This is the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. It allows others to remix and build upon the work non-commercially, as long as they provide credit and license their new creations under the identical terms.",
         "metadata": {"url": "https://creativecommons.org/licenses/by-nc-sa/4.0/", "type": "Copyleft"}},
        {"id": "trademark-use-only-v1",
         "text": "This license applies to brand assets like logos or trademarks. It allows use of the asset only for the promotion of the associated brand or product. It cannot be used on other products or in a way that implies endorsement without permission.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/trademark-use-only-v1", "type": "Trademark"}},
        {"id": "nonexclusive-commercial-v1",
         "text": "This is a non-exclusive commercial license. The asset can be used for business and for-profit purposes, but the same license can also be granted to other licensees.",
         "metadata": {"url": "https://metaverse.myplatform.com/licenses/nonexclusive-commercial-v1",
                      "type": "Commercial"}},
    ]

    try:
        # 1. Initialize the manager to connect to the database.
        vector_db = VectorDBManager(
            collection_name="aigc_governance_policies",
            host="localhost",
            port=8000
        )

        print(f"\nAttempting to delete old collection '{vector_db.collection_name}' to ensure a clean slate...")
        try:
            vector_db.client.delete_collection(name=vector_db.collection_name)
            print(f"Collection '{vector_db.collection_name}' deleted successfully.")
        except Exception:
            # This error is okay. It just means the collection didn't exist.
            print(f"Info: Could not delete collection (it likely did not exist, which is fine).")

        # 3. Now, create the collection cleanly with the correct settings.
        print(f"Creating new collection '{vector_db.collection_name}'...")
        vector_db.collection = vector_db.client.get_or_create_collection(
            name=vector_db.collection_name,
            embedding_function=vector_db.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        print(
            f"Collection '{vector_db.collection_name}' created successfully with the correct embedding model.")

        # 4. Populate the fresh collection with the policies.
        populate_policies(vector_db, all_policies)

        print("\n--- ✅ ChromaDB Setup Complete ✅ ---")
    except Exception as e:
        print(f"\n--- ❌ An error occurred during ChromaDB setup: {e} ❌ ---")
        sys.exit(1)
