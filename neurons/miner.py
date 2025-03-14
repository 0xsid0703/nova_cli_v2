import base64
import os
import math
import random
import argparse
import asyncio
import tempfile
import datetime
from typing import cast
from types import SimpleNamespace
import sys
from dotenv import load_dotenv
import requests
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import get_sequence_from_protein_code, upload_file_to_github
from PSICHIC.wrapper import PsichicWrapper
from btdr import QuicknetBittensorDrandTimelock

CLIENT_ID = 0
CLIENT_SECRET = [[0, 7], [8, 15], [16, 23], [24, 31], [32, 39], [40, 47], [48, 55], [56, 63], [64, 71], [72, 79], [80, 87], [88, 95], [96, 103], [104, 109]]

class Miner:
    def __init__(self):
        load_dotenv()

        self.hugging_face_dataset_repo = 'Metanova/SAVI-2020'
        self.psichic_result_column_name = 'predicted_binding_affinity'
        self.chunk_size = 128
        self.tolerance = 3

        self.config = self.get_config()
        node = SubstrateInterface(url=self.config.network)
        self.epoch_length = node.query("SubtensorModule", "Tempo", [self.config.netuid]).value
        self.setup_logging()
        self.current_block = 0
        self.current_challenge_target = None
        self.last_challenge_target = None
        self.current_challenge_antitarget = None
        self.last_challenge_antitarget = None
        self.psichic_target = PsichicWrapper()
        self.psichic_antitarget = PsichicWrapper()
        self.candidate_product = None
        self.candidate_product_score = 0
        self.best_score = -math.inf
        self.last_submitted_product = None
        self.last_submission_time = None
        self.submission_interval = 1200
        self.inference_task = None
        self.shutdown_event = asyncio.Event()
        self.bdt = QuicknetBittensorDrandTimelock()


    def get_config(self):
        # Set up the configuration parser.
        parser = argparse.ArgumentParser()
        # Adds override arguments for network.
        parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
        # Adds override arguments for network and netuid.
        parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
        # Adds subtensor specific arguments.
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments.
        bt.logging.add_args(parser)
        # Adds wallet specific arguments.
        bt.wallet.add_args(parser)
        # Parse the config.
        config = bt.config(parser)
        # Set up logging directory.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                'miner',
            )
        )
        # Ensure the logging directory exists.
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        # Set up logging.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:")
        bt.logging.info(self.config)

    async def setup_bittensor_objects(self):
        # Build Bittensor validator objects.
        bt.logging.info("Setting up Bittensor objects.")

        # Initialize wallet.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        async with bt.async_subtensor(network=self.config.network) as subtensor:
            self.subtensor = subtensor
            bt.logging.info(f"Subtensor: {self.subtensor}")

            # Initialize and sync metagraph
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            await self.metagraph.sync()
            bt.logging.info(f"Metagraph synced: {self.metagraph}")
            
            # Log stake distribution
            stakes = self.metagraph.S.tolist()
            sorted_stakes = sorted([(i, s) for i, s in enumerate(stakes)], key=lambda x: x[1], reverse=True)
            bt.logging.info("Top 5 validators by stake:")
            for uid, stake in sorted_stakes[:5]:
                bt.logging.info(f"UID: {uid}, Stake: {stake}")

            # Get miner uid

    def stream_random_chunk_from_dataset(self):
        # Streams a random chunk from the dataset repo on huggingface.
        files = list_repo_files(self.hugging_face_dataset_repo, repo_type='dataset')
        files = [file for file in files if file.endswith('.csv')]
        num_files = len(files)  # Get the number of CSV files

        limited_files = files[CLIENT_SECRET[CLIENT_ID][0]:CLIENT_SECRET[CLIENT_ID][1]]
        random_file = random.choice(limited_files)
        dataset_dict = load_dataset(self.hugging_face_dataset_repo,
                                    data_files={'train': random_file},
                                    streaming=True,
                                    )
        dataset = dataset_dict['train']
        batched = dataset.batch(self.chunk_size)
        return batched
    
    async def get_protein_from_epoch_start(self):
        """
        Picks the highest-stake protein from the window [epoch_start .. epoch_start + tolerance].
        """
        url = "https://nova-api-sid.vercel.app/api/getLatestProtein"
        headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
        response = requests.get(url, headers=headers)
        res = response.json()
        pre_protein = res['protein']['protein'].split('|')
        target_protein = pre_protein[0]
        antitarget_protein = pre_protein[1]
        return (target_protein, antitarget_protein)

    async def run_psichic_model_loop(self):
        """
        Continuously runs the PSICHIC model on batches of molecules from the dataset.

        This method streams random chunks of molecule data from a Hugging Face dataset,
        processes them through the PSICHIC model to predict binding affinities, and updates
        the best candidate when a higher scoring molecule is found. Runs in a separate thread
        until the shutdown event is triggered.

        The method:
        1. Streams data in chunks from the dataset
        2. Cleans the product names and SMILES strings
        3. Runs PSICHIC predictions on each chunk for both target and antitarget proteins
        4. Updates the best candidate if a higher score is found
        5. Continues until shutdown_event is set

        Raises:
            Exception: Logs any errors during execution and sets the shutdown event
        """
        dataset = self.stream_random_chunk_from_dataset()
        while not self.shutdown_event.is_set():
            try:
                for chunk in dataset:
                    df = pd.DataFrame.from_dict(chunk)
                    df['product_name'] = df['product_name'].apply(lambda x: x.replace('"', ''))
                    df['product_smiles'] = df['product_smiles'].apply(lambda x: x.replace('"', ''))

                    # Run the PSICHIC model for target and antitarget proteins on the chunk.
                    bt.logging.debug(f'Running inference...')
                    psichic_scores_target = self.psichic_target.run_validation(df['product_smiles'].tolist())
                    psichic_scores_antitarget = self.psichic_antitarget.run_validation(df['product_smiles'].tolist())

                    # Merge the scores for target and antitarget proteins
                    psichic_scores_target.rename(columns={self.psichic_result_column_name: "target_affinity"}, inplace=True)
                    psichic_scores_target['antitarget_affinity'] = psichic_scores_antitarget[self.psichic_result_column_name]
                    psichic_scores_target['affinity_difference'] = psichic_scores_target['target_affinity'] - psichic_scores_target['antitarget_affinity']

                    psichic_scores_target = psichic_scores_target.sort_values(by='affinity_difference', ascending=False).reset_index(drop=True)

                    if psichic_scores_target['affinity_difference'].iloc[0] > self.best_score:
                        candidate_molecule = psichic_scores_target['Ligand'].iloc[0]
                        self.best_score = psichic_scores_target['affinity_difference'].iloc[0]
                        self.candidate_product = df.loc[df['product_smiles'] == candidate_molecule, 'product_name'].iloc[0]
                        bt.logging.info(f"New best score: {self.best_score}, New candidate product: {self.candidate_product}")
                        url = "https://nova-api-sid.vercel.app/api/setLatestScore"
                        data = {"protein": f"{self.current_challenge_target}|{self.current_challenge_antitarget}", "product": self.candidate_product, "score": float(self.best_score)}
                        headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
                        response = requests.post(url, json=data, headers=headers)

                        await asyncio.sleep(1)
                    await asyncio.sleep(1)

            except Exception as e:
                bt.logging.error(f"Error running PSICHIC model: {e}")
                self.shutdown_event.set()

    async def run(self):
        # The Main Mining Loop.
        bt.logging.info("Starting miner loop.")
        await self.setup_bittensor_objects()

        # Startup case: In case we start mid-epoch get most recent protein and start inference
        current_block = await self.subtensor.get_current_block()
        last_boundary = (current_block // self.epoch_length) * self.epoch_length
        start_target, start_antitarget = await self.get_protein_from_epoch_start()
        if start_target and start_antitarget:

            self.current_challenge_target = start_target
            self.last_challenge_target = start_target

            self.current_challenge_antitarget = start_antitarget
            self.last_challenge_antitarget = start_antitarget

            bt.logging.info(f"Startup target: {start_target}, antitarget: {start_antitarget}")

            protein_sequence_target = get_sequence_from_protein_code(start_target)
            protein_sequence_antitarget = get_sequence_from_protein_code(start_antitarget)

            try:
                self.psichic_target.run_challenge_start(protein_sequence_target)
                bt.logging.info(f"Initialized model for {protein_sequence_target}")

                self.psichic_antitarget.run_challenge_start(protein_sequence_antitarget)
                bt.logging.info(f"Initialized model for {protein_sequence_antitarget}")
            except Exception as e:
                bt.logging.error(f"Error initializing model: {e}")

            try:
                self.inference_task = asyncio.create_task(self.run_psichic_model_loop())
                bt.logging.debug("Inference started on startup protein.")
            except Exception as e:
                bt.logging.error(f"Error starting inference: {e}")


        while True:
            try:
                current_block = await self.subtensor.get_current_block()
                # If we are at the epoch boundary, wait for the tolerance blocks to find a new protein
                if current_block % self.epoch_length == 0:
                    bt.logging.info(f"Epoch boundary at block {current_block}, waiting {self.tolerance} blocks.")
                    new_target, new_antitarget = await self.get_protein_from_epoch_start()
                    if (new_target and new_antitarget) and (new_target != self.last_challenge_target or new_antitarget != self.last_challenge_antitarget):
                        self.current_challenge_target = new_target
                        self.last_challenge_target = new_target

                        self.current_challenge_antitarget = new_antitarget
                        self.last_challenge_antitarget = new_antitarget
                        bt.logging.info(f"New proteins: {new_target}, {new_antitarget}")

                    # If old task still running, set shutdown event
                    if self.inference_task:
                        if not self.inference_task.done():
                            self.shutdown_event.set()
                            bt.logging.debug(f"Shutdown event set for old inference task.")

                            # reset old values for best score, etc
                            self.candidate_product = None
                            self.candidate_product_score = 0
                            self.best_score = 0
                            self.last_submitted_product = None
                            self.shutdown_event = asyncio.Event()

                    # Get protein sequence from uniprot
                    protein_sequence_target = get_sequence_from_protein_code(self.current_challenge_target)
                    protein_sequence_antitarget = get_sequence_from_protein_code(self.current_challenge_antitarget)

                    # Initialize PSICHIC for new protein
                    try:
                        bt.logging.info(f'Initializing model for protein sequence: {protein_sequence_target}')
                        self.psichic_target.run_challenge_start(protein_sequence_target)
                        bt.logging.info('Model initialized successfully for target protein.')

                        bt.logging.info(f'Initializing model for protein sequence: {protein_sequence_antitarget}')
                        self.psichic_antitarget.run_challenge_start(protein_sequence_antitarget)
                        bt.logging.info('Model initialized successfully for antitarget protein.')
                    except Exception as e:
                        try:
                            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
                            self.psichic_target.run_challenge_start(protein_sequence_target)
                            bt.logging.info('Model initialized successfully for target protein.')

                            self.psichic_antitarget.run_challenge_start(protein_sequence_antitarget)
                            bt.logging.info('Model initialized successfully for antitarget protein.')
                        except Exception as e:
                            bt.logging.error(f'Error initializing model: {e}')

                    # Start inference loop
                    try:
                        self.inference_task = asyncio.create_task(self.run_psichic_model_loop())
                        bt.logging.debug(f'Inference task started successfully')
                    except Exception as e:
                        bt.logging.error(f'Error initializing inference: {e}')

                await asyncio.sleep(1)

                # Periodically update our knowledge of the network graph.
                if self.current_block % 60 == 0:
                    await self.metagraph.sync()
                    log = (
                        f'Block: {self.metagraph.block.item()} | '
                        f'Number of nodes: {self.metagraph.n} | '
                        f'Current epoch: {self.metagraph.block.item() // self.epoch_length}'
                    )
                    bt.logging.info(log)
                self.current_block += 1

            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting miner.")
                exit()

# Run the miner.
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())

