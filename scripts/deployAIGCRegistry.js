// Import Hardhat runtime environment (hre) - provides access to ethers.js, network config etc.
const hre = require("hardhat");

async function main() {
  console.log("Deploying AIGCRegistry contract...");

  // Get the deployer account (based on the private key configured in hardhat.config.js)
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with the account:", deployer.address);

  // Get balance before deployment (optional, good for sanity check)
  const balanceBigInt = await hre.ethers.provider.getBalance(deployer.address);
  const balanceInEth = hre.ethers.formatEther(balanceBigInt);
  console.log(`Account balance: ${balanceInEth} ETH`);


  // Get the ContractFactory for AIGCRegistry
  // Hardhat-ethers automatically links the contract name to the compiled artifact.
  const AIGCRegistry = await hre.ethers.getContractFactory("AIGCRegistry");

  // Start the deployment process.
  // The constructor of AIGCRegistry takes NO arguments in the latest version.
  // Ownable automatically sets the deployer as the initial owner.
  console.log("Deploying contract (deployer will be initial owner)...");
  // Call deploy() with NO arguments, as the constructor expects none.
  const aigcRegistry = await AIGCRegistry.deploy(); // <-- REMOVED deployer.address argument

  // Wait for the deployment transaction to be mined and the contract to be fully deployed.
  // It's important to wait for the deployment to complete before interacting with the contract
  // or considering the deployment successful.
  // The `waitForDeployment` method replaces the older `deployed()` method.
  await aigcRegistry.waitForDeployment();

  // Get the address the contract was deployed to.
  const contractAddress = await aigcRegistry.getAddress();
  console.log(`AIGCRegistry deployed to: ${contractAddress}`);

  // Optional: Log balance after deployment
  const balanceAfterBigInt = await hre.ethers.provider.getBalance(deployer.address);
  const balanceAfterInEth = hre.ethers.formatEther(balanceAfterBigInt);
  console.log(`Account balance after deployment: ${balanceAfterInEth} ETH`);
  console.log(`Deployment cost (approx): ${parseFloat(balanceInEth) - parseFloat(balanceAfterInEth)} ETH`);

}

// Standard pattern to execute the async main function and handle errors.
main()
  .then(() => process.exit(0)) // Exit script successfully on completion
  .catch((error) => {
    console.error("Deployment failed:", error); // Log any errors
    process.exit(1); // Exit script with error code
  });
