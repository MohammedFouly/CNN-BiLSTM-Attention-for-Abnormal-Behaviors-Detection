const fs = require("fs");
const path = require("path");
const solc = require("solc");

const contractPath = path.join(__dirname, "contracts", "HealthcareAccessRegistry.sol");
const source = fs.readFileSync(contractPath, "utf8");

const input = {
  language: "Solidity",
  sources: {
    "HealthcareAccessRegistry.sol": { content: source },
  },
  settings: {
    viaIR: true,
    optimizer: { enabled: true, runs: 200 },
    outputSelection: {
      "*": {
        "*": ["abi", "evm.bytecode.object"],
      },
    },
  },
};

const output = JSON.parse(solc.compile(JSON.stringify(input)));

if (output.errors) {
  const fatal = output.errors.filter((e) => e.severity === "error");
  output.errors.forEach((e) => console.log(e.formattedMessage));
  if (fatal.length > 0) process.exit(1);
}

const contract = output.contracts["HealthcareAccessRegistry.sol"]["HealthcareAccessRegistry"];

const artifact = {
  abi: contract.abi,
  bytecode: "0x" + contract.evm.bytecode.object,
};

fs.mkdirSync(path.join(__dirname, "build"), { recursive: true });
fs.writeFileSync(
  path.join(__dirname, "build", "HealthcareAccessRegistry.json"),
  JSON.stringify(artifact, null, 2)
);

console.log("Compiled OK, bytecode length:", artifact.bytecode.length);
