import json
import os
import random
import time
from datetime import datetime, timezone

from web3 import Web3

CHAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chain")
RPC_URL = "http://127.0.0.1:8545"

ROLES = ["physician", "nurse", "billing_clerk", "lab_tech", "insurance_auditor", "admin"]
ACTIONS = ["record_access", "prescription_write", "insurance_claim", "consent_update", "credential_check", "lab_result_upload"]
HIGH_RISK_ROLES = {"billing_clerk", "admin"}


class ChainClient:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        with open(os.path.join(CHAIN_DIR, "deployment.json")) as f:
            deployment = json.load(f)
        self.address = deployment["address"]
        self.abi = deployment["abi"]
        self.chain_id = deployment["chain_id"]
        self.contract = self.w3.eth.contract(address=self.address, abi=self.abi)
        self.accounts = self.w3.eth.accounts
        self.last_processed_block = self.w3.eth.block_number
        self.gas_price_history = []
        self.rng = random.Random(7)

    def is_connected(self):
        return self.w3.is_connected()

    def latest_block_number(self):
        return self.w3.eth.block_number

    def submit_random_event(self, anomaly_rate=0.1):
        account = self.rng.choice(self.accounts)
        role = self.rng.choice(ROLES)
        action = self.rng.choice(ACTIONS)
        is_anom = self.rng.random() < anomaly_rate

        sensitive = self.rng.random() < 0.25
        role_mismatch = False
        cred_revocation = False
        geoloc = False
        collusion = False
        access_freq = self.rng.randint(0, 5)
        session_dur = self.rng.randint(30, 900)

        if is_anom:
            sensitive = True
            role_mismatch = role not in HIGH_RISK_ROLES and self.rng.random() < 0.7
            cred_revocation = self.rng.random() < 0.5
            geoloc = self.rng.random() < 0.3
            collusion = self.rng.random() < 0.15
            access_freq = self.rng.randint(20, 50)
            session_dur = self.rng.randint(1, 20)

        gas_price = self.w3.eth.gas_price
        if is_anom:
            gas_price = int(gas_price * self.rng.uniform(4, 9))

        tx_hash = self.contract.functions.recordEvent(
            role, action, sensitive, role_mismatch, cred_revocation,
            geoloc, collusion, access_freq, session_dur,
        ).transact({"from": account, "gasPrice": gas_price})

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt

    def poll_new_events(self):
        latest = self.w3.eth.block_number
        if latest <= self.last_processed_block:
            return []

        logs = self.contract.events.HealthcareEvent().get_logs(
            from_block=self.last_processed_block + 1, to_block=latest
        )
        results = []
        for log in logs:
            args = log["args"]
            block = self.w3.eth.get_block(log["blockNumber"])
            tx = self.w3.eth.get_transaction(log["transactionHash"])

            gas_price = tx["gasPrice"]
            self.gas_price_history.append(gas_price)
            if len(self.gas_price_history) > 200:
                self.gas_price_history.pop(0)
            baseline = sum(self.gas_price_history) / len(self.gas_price_history)
            gas_deviation = abs(gas_price - baseline) / baseline if baseline > 0 else 0.0

            dt = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc)
            input_size = len(tx["input"]) if tx["input"] else 0

            results.append({
                "tx_hash": log["transactionHash"].hex(),
                "block_number": log["blockNumber"],
                "entity": args["entity"],
                "role": args["role"],
                "action": args["action"],
                "sensitive_record_accessed": args["sensitiveRecordAccessed"],
                "role_mismatch": args["roleMismatch"],
                "credential_revocation": args["credentialRevocation"],
                "geolocation_mismatch": args["geolocationMismatch"],
                "issuer_verifier_collusion": args["issuerVerifierCollusion"],
                "access_frequency": args["accessFrequency"],
                "session_duration": args["sessionDurationSeconds"],
                "gas_price": gas_price,
                "gas_deviation": gas_deviation,
                "input_size": input_size,
                "hour_of_day": dt.hour,
                "timestamp": block["timestamp"],
            })

        self.last_processed_block = latest
        return results


def event_to_feature_vector(ev):
    from .onchain_model import FEATURE_ORDER
    features = {
        "hour_of_day": ev["hour_of_day"] / 23.0,
        "time_since_last_access": 0.0,
        "access_frequency": min(ev["access_frequency"] / 20.0, 1.0),
        "avg_session_duration": min(ev["session_duration"] / 900.0, 1.0),
        "gas_price_deviation": min(ev["gas_deviation"], 1.0),
        "input_data_size": min(ev["input_size"] / 2000.0, 1.0),
        "sensitive_record_accessed": 1.0 if ev["sensitive_record_accessed"] else 0.0,
        "role_mismatch": 1.0 if ev["role_mismatch"] else 0.0,
        "credential_revocation": 1.0 if ev["credential_revocation"] else 0.0,
        "geolocation_mismatch": 1.0 if ev["geolocation_mismatch"] else 0.0,
        "issuer_verifier_collusion": 1.0 if ev["issuer_verifier_collusion"] else 0.0,
    }
    return [features[f] for f in FEATURE_ORDER]
