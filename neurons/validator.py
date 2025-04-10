import asyncio
from ast import literal_eval
import math
import os
import sys
import argparse
import binascii
from typing import cast
from types import SimpleNamespace
import bittensor as bt
from substrateinterface import SubstrateInterface
import requests
from dotenv import load_dotenv
from bittensor.core.chain_data.utils import decode_metadata

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import get_smiles, get_index_in_range_from_blockhash, get_protein_code_at_index, get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper
from btdr import QuicknetBittensorDrandTimelock

psichic = PsichicWrapper()
btd = QuicknetBittensorDrandTimelock()

def get_config():
    """
    Parse command-line arguments to set up the configuration for the wallet
    and subtensor client.
    """
    load_dotenv()
    parser = argparse.ArgumentParser('Nova')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    config = bt.config(parser)
    config.netuid = 68
    config.network = os.environ.get("SUBTENSOR_NETWORK")
    node = SubstrateInterface(url=config.network)
    config.epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value

    return config

async def check_registration(wallet, subtensor, netuid):
    """
    Confirm that the wallet hotkey is in the metagraph for the specified netuid.
    Logs an error and exits if it's not registered. Warns if stake is less than 1000.
    """
    metagraph = await subtensor.metagraph(netuid=netuid)
    my_hotkey_ss58 = wallet.hotkey.ss58_address

    if my_hotkey_ss58 not in metagraph.hotkeys:
        bt.logging.error(f"Hotkey {my_hotkey_ss58} is not registered on netuid {netuid}.")
        bt.logging.error("Are you sure you've registered and staked?")
        sys.exit(1) 
    
    uid = metagraph.hotkeys.index(my_hotkey_ss58)
    myStake = metagraph.S[uid]
    bt.logging.info(f"Hotkey {my_hotkey_ss58} found with UID={uid} and stake={myStake}")

    if (myStake < 1000):
        bt.logging.warning(f"Hotkey has less than 1000 stake, unable to validate")

def run_model(protein: str, molecule: str) -> float:
    """
    Given a protein sequence (protein) and a molecule identifier (molecule),
    retrieves its SMILES string, then uses the PsichicWrapper to produce
    a predicted binding score. Returns 0.0 if SMILES not found or if
    there's any issue with scoring.
    """

    # Initialize PSICHIC for new protein
    bt.logging.info(f'Initializing model for protein sequence: {protein}')
    try:
        psichic.run_challenge_start(protein)
        bt.logging.info('Model initialized successfully.')
    except Exception as e:
        try:
            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
            psichic.run_challenge_start(protein)
            bt.logging.info('Model initialized successfully.')
        except Exception as e:
            bt.logging.error(f'Error initializing model: {e}')


    smiles = get_smiles(molecule)
    if not smiles:
        bt.logging.debug(f"Could not retrieve SMILES for '{molecule}', returning score of 0.0.")
        return 0.0

    results_df = psichic.run_validation([smiles])  # returns a DataFrame
    if results_df.empty:
        bt.logging.warning("Psichic returned an empty DataFrame, returning 0.0.")
        return 0.0

    predicted_score = results_df.iloc[0]['predicted_binding_affinity']
    if predicted_score is None:
        bt.logging.warning("No 'predicted_binding_affinity' found, returning 0.0.")
        return 0.0

    return float(predicted_score)

def run_model_difference(target_sequence: str, antitarget_sequence: str, molecule: str) -> float:
    """
    Compute final_score = binding_affinity(target) - binding_affinity(anti-target)
    """
    s_target = run_model(protein=target_sequence, molecule=molecule)
    s_anti   = run_model(protein=antitarget_sequence, molecule=molecule)
    return s_target - s_anti


async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int) -> dict:
    """
    Retrieve commitments for all miners on a given subnet (netuid) at a specific block.

    Args:
        subtensor: The subtensor client object.
        netuid (int): The network ID.
        block (int, optional): The block number to query. Defaults to None.

    Returns:
        dict: A mapping from hotkey to a SimpleNamespace containing uid, hotkey,
              block, and decoded commitment data.
    """

    # Gather commitment queries for all validators (hotkeys) concurrently.
    commits = await asyncio.gather(*[
        subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
        ) for hotkey in metagraph.hotkeys
    ])

    # Process the results and build a dictionary with additional metadata.
    result = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        commit = cast(dict, commits[uid])
        if commit:
            result[hotkey] = SimpleNamespace(
                uid=uid,
                hotkey=hotkey,
                block=commit['block'],
                data=decode_metadata(commit)
            )
    return result
def decrypt_submissions(current_commitments: dict, headers: dict = {"Range": "bytes=0-1024"}) -> dict:
    """
    Decrypts submissions from validators by fetching encrypted content from GitHub URLs and decrypting them.

    Args:
        current_commitments (dict): A dictionary of miner commitments where each value contains:
            - uid: Miner's unique identifier
            - data: GitHub URL path containing the encrypted submission
            - Other commitment metadata
        headers (dict, optional): HTTP request headers for fetching content. 
            Defaults to {"Range": "bytes=0-2048"} to limit response size.

    Returns:
        dict: A dictionary of decrypted submissions mapped by validator UIDs.
            Empty if no valid submissions were found or decryption failed.

    Note:
        - Only processes commitments where data contains a '/' (indicating a GitHub URL)
        - Uses btd.decrypt_dict for decryption of the fetched submissions
        - Logs errors for failed HTTP requests and submission counts
    """
    encrypted_submissions = {}
    for commit in current_commitments.values():
        if '/' in commit.data: # Filter only url submissions
            full_url = f"https://raw.githubusercontent.com/{commit.data}"
            response = requests.get(full_url, headers=headers)
            if response.status_code in [200, 206]:
                encrypted_content = response.content
                encrypted_content = encrypted_content.decode('utf-8', errors='replace')
                encrypted_content = literal_eval(encrypted_content)
                if type(encrypted_content) != tuple:
                    bt.logging.error(f"Encrypted content for {commit.uid} is not a tuple: {encrypted_content}")
                    continue
                encrypted_submissions[commit.uid] = (encrypted_content[0], encrypted_content[1])
            else:
                bt.logging.error(f"Error fetching encrypted submission: {response.status_code}")
                continue

    bt.logging.info(f"Encrypted submissions: {len(encrypted_submissions)}")
    
    decrypted_submissions = btd.decrypt_dict(encrypted_submissions)
    bt.logging.info(f"Decrypted submissions: {len(decrypted_submissions)}")
            
    return decrypted_submissions


async def main(config):
    """
    Main routine that continuously checks for the end of an epoch to perform:
        - Setting a new commitment.
        - Retrieving past commitments.
        - Selecting the best protein/molecule pairing based on stakes and scores.
        - Setting new weights accordingly.

    Args:
        config: Configuration object for subtensor and wallet.
    """
    wallet = bt.wallet(config=config)

    # Initialize the asynchronous subtensor client.
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    # Check if the hotkey is registered and has at least 1000 stake.
    await check_registration(wallet, subtensor, config.netuid)

    tolerance = 3 # block tolerance window for validators to commit protein

    while True:
        # Fetch the current metagraph for the given subnet (netuid 68).
        metagraph = await subtensor.metagraph(config.netuid)
        bt.logging.debug(f'Found {metagraph.n} nodes in network')
        current_block = await subtensor.get_current_block()

        # Check if the current block marks the end of an epoch (using a 360-block interval).
        if current_block % config.epoch_length == 0:

            try:
                current_block_hash = await subtensor.determine_block_hash(current_block)
                prev_block_hash = await subtensor.determine_block_hash(current_block - 1)

                target_random_index = get_index_in_range_from_blockhash(current_block_hash, 179620)
                antitarget_random_index = get_index_in_range_from_blockhash(prev_block_hash, 179620)

                target_protein_code = get_protein_code_at_index(target_random_index)
                antitarget_protein_code = get_protein_code_at_index(antitarget_random_index)

                await subtensor.set_commitment(
                    wallet=wallet,
                    netuid=config.netuid,
                    data=f"{target_protein_code}|{antitarget_protein_code}"
                )
                bt.logging.info(f"Committed successfully target: {target_protein_code}, antitarget: {antitarget_protein_code}")

            except Exception as e:
                bt.logging.error(f"Error: {e}")
            # Retrieve commitments from the previous epoch.
            prev_epoch = current_block - config.epoch_length
            best_stake = -math.inf
            current_protein = None

            block_to_check = prev_epoch
            block_hash_to_check = await subtensor.determine_block_hash(block_to_check + tolerance)  
            epoch_metagraph = await subtensor.metagraph(config.netuid, block=block_to_check + tolerance)
            epoch_commitments = await get_commitments(subtensor, epoch_metagraph, block_hash_to_check, netuid=config.netuid)
            epoch_commitments = {k: v for k, v in epoch_commitments.items() if current_block - v.block <= (config.epoch_length + tolerance)}
            
            high_stake_protein_commitment = max(
                epoch_commitments.values(),
                key=lambda commit: epoch_metagraph.S[commit.uid],
                default=None
            )
            if not high_stake_protein_commitment:
                bt.logging.error("Error getting current protein commitment.")
                current_protein = None
                continue

            protein_codes = high_stake_protein_commitment.data.split('|')
            target_protein_code = protein_codes[0]
            antitarget_protein_code = protein_codes[1]
            bt.logging.info(f"Current target protein: {target_protein_code}, antitarget: {antitarget_protein_code}")

            target_protein_sequence = get_sequence_from_protein_code(target_protein_code)
            antitarget_protein_sequence = get_sequence_from_protein_code(antitarget_protein_code)

            # Retrieve the latest commitments (current epoch).
            current_block_hash = await subtensor.determine_block_hash(current_block)
            current_commitments = await get_commitments(subtensor, metagraph, current_block_hash, netuid=config.netuid)
            bt.logging.debug(f"Current commitments: {len(list(current_commitments.values()))}")

             # Decrypt submissions
            decrypted_submissions = decrypt_submissions(current_commitments)

            # Identify the best molecule based on the scoring function.
            best_score = -math.inf
            total_commits = 0
            best_molecule = None
            for hotkey, commit in current_commitments.items():
                if current_block - commit.block <= config.epoch_length:
                    # Find the decrypted submission for the current commitment
                    try:    
                        molecule = decrypted_submissions[commit.uid]
                        total_commits += 1
                    except Exception as e:
                        bt.logging.error(f"Decrypted submission for {commit.uid} not found: {e}")
                        continue
                    score = run_model_difference(target_protein_sequence, antitarget_protein_sequence, commit.data)
                    score = round(score, 3)
                    # If the score is higher, or equal but the block is earlier, update the best.
                    if (score > best_score) or (score == best_score and best_molecule is not None and commit.block < best_molecule.block):
                        best_score = score
                        best_molecule = commit

            # Ensure a best molecule was found before setting weights.
            if best_molecule is not None:
                try:
                    # Create weights where the best molecule's UID receives full weight.
                    weights = [0.0 for i in range(metagraph.n)]
                    print(current_block)
                    weights[best_molecule.uid] = 1.0
                    print(weights)
                    uids = list(range(metagraph.n))
                    result, message = await subtensor.set_weights(
                        wallet=wallet,
                        uids=uids,
                        weights=weights,
                        netuid=config.netuid,
                        wait_for_inclusion=True,
                        )
                    if result:
                        bt.logging.info(f"Weights set successfully: {weights}.")
                    else:
                        bt.logging.error(f"Error setting weights: {message}")
                except Exception as e:
                    bt.logging.error(f"Error setting weights: {e}")
            else:
                bt.logging.info("No valid molecule commitment found for current epoch.")

            # Sleep briefly to prevent busy-waiting (adjust sleep time as needed).
            await asyncio.sleep(1)
            
        # keep validator alive
        elif current_block % (config.epoch_length/2) == 0:
            subtensor = bt.async_subtensor(network=config.network)
            await subtensor.initialize()
            bt.logging.info("Validator reset subtensor connection.")
            await asyncio.sleep(12) # Sleep for 1 block to avoid unncessary re-connection
            
        else:
            bt.logging.info(f"Waiting for epoch to end... {config.epoch_length - (current_block % config.epoch_length)} blocks remaining.")
            await asyncio.sleep(1)


if __name__ == "__main__":
    config = get_config()
    asyncio.run(main(config))
