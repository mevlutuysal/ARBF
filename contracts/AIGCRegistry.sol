// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28; // Use a recent Solidity version

// Import OpenZeppelin contracts for ERC721 standard and utility functions
import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol"; // Imports Ownable functionality
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title AIGCRegistry
 * @dev An ERC721 Non-Fungible Token contract to register AI-Generated Content (AIGC).
 * It stores a standard token URI (pointing to metadata JSON) and an additional
 * Content Identifier (CAR CID) directly on-chain for enhanced verifiability.
 * The CAR CID points to a Content Addressable aRchive file containing the
 * AIGC and its cryptographic hashes, typically stored on IPFS.
 * The deployer of the contract is automatically set as the initial owner.
 */
contract AIGCRegistry is ERC721, ERC721URIStorage, Ownable { // Inherit Ownable
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIdCounter;

    // Mapping from token ID to the CAR file Content Identifier (CID) stored as bytes32
    mapping(uint256 => bytes32) private _carFileCIDs;

    // Event emitted when a new token is registered along with its CAR CID
    event TokenRegistered(
        uint256 indexed tokenId,
        address indexed owner,
        string tokenURI,
        bytes32 carCID
    );

    /**
     * @dev Constructor initializes the ERC721 token name and symbol.
     * It also initializes Ownable, setting the deployer (msg.sender) as the initial owner.
     */
    constructor()
        ERC721("AI Generated Content Registry", "AIGCR")
        // Ownable() is implicitly called here, setting msg.sender as owner
        // No need to explicitly pass initialOwner if using default behavior
    {
        // Transfer ownership immediately if a different initial owner is desired
        // transferOwnership(initialOwner); // Uncomment and pass address if needed
    }

    /**
     * @dev Mints a new token, assigns it to the `recipient`, sets its metadata URI,
     * and stores the associated CAR file CID.
     * Can only be called by the contract owner.
     * Emits a {TokenRegistered} event.
     * @param recipient The address that will receive the minted NFT.
     * @param _tokenURI The URI pointing to the off-chain metadata JSON file. Renamed to avoid shadowing.
     * @param carCID The bytes32 representation of the CAR file CID.
     */
    function safeMintWithCARCID(address recipient, string memory _tokenURI, bytes32 carCID)
        public
        onlyOwner // Restrict minting to the owner
    {
        require(recipient != address(0), "AIGCRegistry: Mint to the zero address");
        require(carCID != bytes32(0), "AIGCRegistry: CAR CID cannot be empty");

        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();

        _safeMint(recipient, tokenId);
        _setTokenURI(tokenId, _tokenURI); // Use the renamed parameter _tokenURI
        _carFileCIDs[tokenId] = carCID;

        emit TokenRegistered(tokenId, recipient, _tokenURI, carCID); // Emit with renamed parameter
    }

    /**
     * @dev Returns the CAR file CID associated with a given token ID.
     */
    function getCarCID(uint256 tokenId) public view returns (bytes32) {
        require(_exists(tokenId), "AIGCRegistry: Token ID does not exist");
        return _carFileCIDs[tokenId];
    }

    // --- Override required functions ---

    /**
     * @dev See {IERC721Metadata-tokenURI}.
     */
    function tokenURI(uint256 tokenId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        require(_exists(tokenId), "AIGCRegistry: URI query for nonexistent token");
        return super.tokenURI(tokenId);
    }

    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }

    /**
     * @dev Hook that is called before any token transfer, including minting and burning.
     * We must override `_burn` because both ERC721 and ERC721URIStorage define it.
     */
    function _burn(uint256 tokenId) internal virtual override(ERC721, ERC721URIStorage) {
        // Call the _burn function from the parent contract (ERC721URIStorage handles URI cleanup)
        super._burn(tokenId);
        // Delete the CAR CID associated with the burned token
        delete _carFileCIDs[tokenId];
    }
}
