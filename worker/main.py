"""
Centrifuge Worker
Asynchronously processes queued data cleaning runs.
"""

import os
import sys
import asyncio
import signal
import logging
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import uuid

# add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_database, close_database, RunManager, CanonicalMappingCache, ArtifactStore
from core.pipeline import DataPipeline
from core.storage import StorageBackend, get_storage_backend

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Worker:
    """
    Worker process that claims and executes data cleaning runs.
    """

    def __init__(self, worker_id: Optional[str] = None):
        """
        Initialize worker.

        Args:
            worker_id: Worker identifier (auto-generated if not provided)
        """
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.db = None
        self.run_manager = None
        self.cache = None
        self.artifact_store = None
        self.storage = None
        self.pipeline = None
        self.running = False
        self.current_run_id = None
        self.heartbeat_interval = int(os.getenv('HEARTBEAT_INTERVAL', '30'))
        self.visibility_timeout = int(os.getenv('VISIBILITY_TIMEOUT', '300'))
        self.poll_interval = int(os.getenv('POLL_INTERVAL', '5'))

    async def initialize(self):
        """Initialize worker components."""
        logger.info(f"Initializing worker {self.worker_id}")

        # initialize database
        self.db = await get_database()
        self.run_manager = RunManager(self.db)
        self.cache = CanonicalMappingCache(self.db)
        self.artifact_store = ArtifactStore(self.db)

        # initialize storage
        self.storage = await get_storage_backend()

        # initialize pipeline
        self.pipeline = DataPipeline(
            storage=self.storage,
            cache=self.cache,
            artifact_store=self.artifact_store
        )

        logger.info(f"Worker {self.worker_id} initialized")

    async def shutdown(self):
        """Shutdown worker components."""
        logger.info(f"Shutting down worker {self.worker_id}")

        self.running = False

        # release current run if any
        if self.current_run_id:
            try:
                await self.run_manager.complete_run(
                    self.current_run_id,
                    'failed',
                    {},
                    error_message="Worker shutdown",
                    error_code="WORKER_SHUTDOWN"
                )
            except Exception as e:
                logger.error(f"Failed to release run on shutdown: {e}")

        # close connections
        if self.storage:
            await self.storage.close()

        await close_database()

        logger.info(f"Worker {self.worker_id} shutdown complete")

    async def heartbeat_loop(self):
        """Send periodic heartbeats for current run."""
        while self.running:
            if self.current_run_id:
                try:
                    success = await self.run_manager.update_heartbeat(
                        self.current_run_id,
                        self.worker_id
                    )
                    if not success:
                        logger.warning(f"Failed to update heartbeat for run {self.current_run_id}")
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")

            await asyncio.sleep(self.heartbeat_interval)

    async def process_run(self, run_data: Dict[str, Any]):
        """
        Process a single run.

        Args:
            run_data: Run details from database
        """
        run_id = run_data['run_id']
        self.current_run_id = run_id

        logger.info(f"Processing run {run_id}")

        try:
            # update progress - starting
            await self.run_manager.update_progress(run_id, 'initializing', 0)

            # check for cancellation
            if await self.run_manager.check_cancel_requested(run_id):
                logger.info(f"Run {run_id} cancelled")
                await self.run_manager.complete_run(
                    run_id, 'failed', {},
                    error_message="Run cancelled",
                    error_code="CANCELLED"
                )
                return

            # execute pipeline phases
            options = run_data['options']
            input_hash = run_data['input_hash']

            # download input file from storage
            input_path = f"inputs/{input_hash[:2]}/{input_hash}/input.csv"
            local_input = f"/tmp/{run_id}_input.csv"

            logger.info(f"Downloading input from {input_path}")
            await self.storage.download(input_path, local_input)

            # phase 1: ingest
            await self.run_manager.update_progress(run_id, 'ingest', 10)
            ingest_result = await self.pipeline.ingest(
                local_input,
                options.get('schema_version', 'v1'),
                use_inference=options.get('use_inference', False)
            )

            if await self.run_manager.check_cancel_requested(run_id):
                return

            # phase 2: deterministic rules
            await self.run_manager.update_progress(run_id, 'rules', 30)
            rules_result = await self.pipeline.apply_rules(ingest_result, run_id)

            if await self.run_manager.check_cancel_requested(run_id):
                return

            # phase 3: residual planning
            await self.run_manager.update_progress(run_id, 'planning', 40)
            plan_result = await self.pipeline.plan_residuals(
                rules_result,
                llm_columns=options.get('llm_columns', ['Department', 'Account Name'])
            )

            if await self.run_manager.check_cancel_requested(run_id):
                return

            # phase 4: llm processing
            await self.run_manager.update_progress(run_id, 'llm', 60)
            llm_result = await self.pipeline.process_llm(
                plan_result,
                dry_run=options.get('dry_run', False)
            )

            if await self.run_manager.check_cancel_requested(run_id):
                return

            # phase 5: final validation
            await self.run_manager.update_progress(run_id, 'validation', 80)
            final_result = await self.pipeline.validate_final(llm_result)

            if await self.run_manager.check_cancel_requested(run_id):
                return

            # phase 6: generate artifacts
            await self.run_manager.update_progress(run_id, 'artifacts', 90)
            artifacts = await self.pipeline.generate_artifacts(
                run_id,
                final_result
            )

            # calculate final metrics
            metrics = {
                'total_rows': final_result.total_rows,
                'cleaned_rows': final_result.cleaned_rows,
                'quarantined_rows': final_result.quarantined_rows,
                'rules_fixed': final_result.rules_fixed_count,
                'llm_fixed': final_result.llm_fixed_count,
                'error_breakdown': final_result.error_breakdown,
                'artifacts': artifacts
            }

            # determine final state
            if final_result.quarantined_rows == 0:
                state = 'succeeded'
            elif final_result.cleaned_rows > 0:
                state = 'partial'
            else:
                state = 'failed'

            # complete run
            await self.run_manager.complete_run(
                run_id,
                state,
                metrics
            )

            logger.info(f"Run {run_id} completed with state {state}")

        except Exception as e:
            logger.error(f"Error processing run {run_id}: {e}", exc_info=True)

            await self.run_manager.complete_run(
                run_id,
                'failed',
                {},
                error_message=str(e),
                error_code="PROCESSING_ERROR"
            )

        finally:
            self.current_run_id = None

            # cleanup temp files
            if os.path.exists(local_input):
                os.remove(local_input)

    async def work_loop(self):
        """Main work loop - claim and process runs."""
        self.running = True

        while self.running:
            try:
                # try to claim a run
                run_data = await self.run_manager.claim_run(
                    self.worker_id,
                    self.visibility_timeout
                )

                if run_data:
                    # process the run
                    await self.process_run(run_data)
                else:
                    # no runs available, wait before polling again
                    await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in work loop: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def run(self):
        """Run the worker."""
        try:
            # initialize components
            await self.initialize()

            # start heartbeat task
            heartbeat_task = asyncio.create_task(self.heartbeat_loop())

            # start work loop
            await self.work_loop()

            # cancel heartbeat
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        finally:
            await self.shutdown()


def handle_signal(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    asyncio.create_task(shutdown())


async def shutdown():
    """Graceful shutdown."""
    logger.info("Initiating graceful shutdown")

    # cancel all tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)

    loop = asyncio.get_event_loop()
    loop.stop()


async def main():
    """Main entry point."""
    logger.info("Starting Centrifuge Worker")

    # setup signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # create and run worker
    worker = Worker()

    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
