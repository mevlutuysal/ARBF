require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config(); // Load environment variables from .env file

// Retrieve environment variables securely
const SEPOLIA_RPC_URL = process.env.SEPOLIA_RPC_URL;
const AGENT_PRIVATE_KEY = process.env.AGENT_PRIVATE_KEY;

// Basic validation
if (!SEPOLIA_RPC_URL) {
  console.error("Missing SEPOLIA_RPC_URL in .env file");
  process.exit(1); // Exit if RPC URL is missing
}
if (!AGENT_PRIVATE_KEY) {
  console.error("Missing AGENT_PRIVATE_KEY in .env file");
  process.exit(1); // Exit if private key is missing
}
//if (!AGENT_PRIVATE_KEY.startsWith('0x')) {
 //console.warn("AGENT_PRIVATE_KEY does not start with 0x. Ensure it's a valid hex private key.");
 // Consider adding more robust validation if needed
//}


/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.28", // Match the pragma version in your contract
    settings: {
      optimizer: {
        enabled: true, // Enable optimizer for gas savings
        runs: 200,     // Standard optimization runs
      },
    },
  },
  networks: {
    // Configuration for the Sepolia test network
    sepolia: {
      url: SEPOLIA_RPC_URL, // RPC endpoint from your .env file
      accounts: [`0x${AGENT_PRIVATE_KEY.replace(/^0x/, '')}`], // Account private key from .env (ensure it has the 0x prefix for ethers)
      chainId: 11155111, // Sepolia's chain ID
    },
    // Optional: Configuration for local development network
    localhost: {
      url: "http://127.0.0.1:8545", // Default Hardhat Network RPC
      chainId: 31337, // Default Hardhat Network chain ID
      // Accounts are automatically provided by Hardhat Network
    },
  },
  etherscan: {
    // Optional: Add API key for contract verification on Etherscan/Basescan etc.
    // apiKey: process.env.ETHERSCAN_API_KEY // Get an API key from https://etherscan.io
  },
  paths: {
    sources: "./contracts", // Location of Solidity source files
    tests: "./test",         // Location of test files
    cache: "./cache",       // Location of Hardhat cache
    artifacts: "./artifacts", // Location of compiled contract artifacts (ABI, bytecode)
  },
  mocha: {
    timeout: 40000 // Increase timeout for potentially long-running tests/deployments
  }
};
