import json
import os

from web3 import Web3

CHAIN_DIR = os.path.dirname(__file__)
RPC_URL = "http://127.0.0.1:8545"


def deploy():
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Could not connect to chain at {RPC_URL}")

    artifact_path = os.path.join(CHAIN_DIR, "build", "HealthcareAccessRegistry.json")
    with open(artifact_path) as f:
        artifact = json.load(f)

    account = w3.eth.accounts[0]
    Contract = w3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])
    tx_hash = Contract.constructor().transact({"from": account})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    deployment = {
        "address": receipt.contractAddress,
        "abi": artifact["abi"],
        "deployer": account,
        "chain_id": w3.eth.chain_id,
        "block_number": receipt.blockNumber,
    }

    with open(os.path.join(CHAIN_DIR, "deployment.json"), "w") as f:
        json.dump(deployment, f, indent=2)

    print(f"Deployed HealthcareAccessRegistry at {receipt.contractAddress} "
          f"(chain id {w3.eth.chain_id}, block {receipt.blockNumber})")
    return deployment


if __name__ == "__main__":
    deploy()
