"""
Concurrency tests for worker claiming mechanism.
Tests atomic run claiming with multiple workers.
"""

import asyncio
import pytest
import uuid
import json
from datetime import datetime, timedelta
import sys
import os

# add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import DatabasePool, RunManager


class TestWorkerConcurrency:
    """Test concurrent worker operations."""
    
    @pytest.fixture
    async def db_pool(self):
        """Create test database pool."""
        pool = DatabasePool()
        await pool.initialize(
            host=os.getenv('TEST_POSTGRES_HOST', 'localhost'),
            database=os.getenv('TEST_POSTGRES_DB', 'centrifuge_test')
        )
        
        # clean up test runs
        await pool.execute("DELETE FROM runs WHERE schema_version = 'test'")
        
        yield pool
        
        # cleanup
        await pool.execute("DELETE FROM runs WHERE schema_version = 'test'")
        await pool.close()
    
    @pytest.fixture
    async def run_manager(self, db_pool):
        """Create run manager."""
        return RunManager(db_pool)
    
    async def create_test_runs(self, run_manager, count: int):
        """Create test runs."""
        run_ids = []
        for i in range(count):
            run_id = await run_manager.create_run(
                input_hash=f"test_hash_{i}",
                options={"test": True},
                schema_version="test"
            )
            run_ids.append(run_id)
        return run_ids
    
    @pytest.mark.asyncio
    async def test_atomic_claiming(self, run_manager):
        """Test that claiming is atomic - only one worker can claim a run."""
        # create test runs
        run_ids = await self.create_test_runs(run_manager, 5)
        
        # simulate multiple workers trying to claim
        async def worker_claim(worker_id: str):
            """Simulate a worker claiming runs."""
            claimed = []
            for _ in range(10):  # try to claim up to 10 runs
                run_data = await run_manager.claim_run(worker_id)
                if run_data:
                    claimed.append(run_data['run_id'])
                else:
                    break
            return claimed
        
        # run multiple workers concurrently
        workers = [f"worker_{i}" for i in range(10)]
        results = await asyncio.gather(*[
            worker_claim(worker_id) for worker_id in workers
        ])
        
        # verify each run was claimed exactly once
        all_claimed = []
        for worker_claims in results:
            all_claimed.extend(worker_claims)
        
        assert len(all_claimed) == len(run_ids)
        assert len(set(all_claimed)) == len(run_ids)  # no duplicates
        
        # verify all runs are now in 'running' state
        for run_id in run_ids:
            status = await run_manager.get_run_status(run_id)
            assert status['state'] == 'running'
    
    @pytest.mark.asyncio
    async def test_visibility_timeout_reclaim(self, run_manager, db_pool):
        """Test that stale runs can be reclaimed after visibility timeout."""
        # create a test run
        run_id = await run_manager.create_run(
            input_hash="test_hash_timeout",
            options={"test": True},
            schema_version="test"
        )
        
        # worker 1 claims the run
        worker1_data = await run_manager.claim_run("worker_1", visibility_timeout=2)
        assert worker1_data is not None
        assert worker1_data['run_id'] == run_id
        
        # worker 2 tries to claim immediately - should get nothing
        worker2_data = await run_manager.claim_run("worker_2", visibility_timeout=2)
        assert worker2_data is None
        
        # simulate stale run by backdating heartbeat
        await db_pool.execute("""
            UPDATE runs 
            SET heartbeat_at = NOW() - INTERVAL '3 seconds'
            WHERE run_id = $1::uuid
        """, run_id)
        
        # worker 2 should now be able to reclaim
        worker2_data = await run_manager.claim_run("worker_2", visibility_timeout=2)
        assert worker2_data is not None
        assert worker2_data['run_id'] == run_id
        
        # verify ownership changed
        status = await run_manager.get_run_status(run_id)
        assert status['state'] == 'running'
    
    @pytest.mark.asyncio
    async def test_heartbeat_prevents_reclaim(self, run_manager):
        """Test that regular heartbeats prevent run reclaim."""
        # create a test run
        run_id = await run_manager.create_run(
            input_hash="test_hash_heartbeat",
            options={"test": True},
            schema_version="test"
        )
        
        # worker 1 claims the run
        worker1_data = await run_manager.claim_run("worker_1", visibility_timeout=2)
        assert worker1_data is not None
        
        # simulate heartbeat updates
        async def heartbeat_loop(run_id, worker_id):
            """Simulate heartbeat updates."""
            for _ in range(5):
                await asyncio.sleep(0.5)
                success = await run_manager.update_heartbeat(run_id, worker_id)
                assert success
        
        # run heartbeat in background
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(run_id, "worker_1")
        )
        
        # wait and try to reclaim multiple times
        for _ in range(3):
            await asyncio.sleep(1)
            worker2_data = await run_manager.claim_run("worker_2", visibility_timeout=2)
            assert worker2_data is None  # should not be able to reclaim
        
        # cancel heartbeat
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_concurrent_progress_updates(self, run_manager):
        """Test concurrent progress updates don't cause conflicts."""
        # create a test run
        run_id = await run_manager.create_run(
            input_hash="test_hash_progress",
            options={"test": True},
            schema_version="test"
        )
        
        # claim the run
        await run_manager.claim_run("worker_1")
        
        # simulate concurrent progress updates
        async def update_progress(phase: str, percent: int):
            """Update progress."""
            await run_manager.update_progress(
                run_id,
                phase,
                percent,
                metrics={"phase": phase, "percent": percent}
            )
        
        # run multiple updates concurrently
        updates = [
            update_progress("phase1", 10),
            update_progress("phase2", 20),
            update_progress("phase3", 30),
            update_progress("phase4", 40),
            update_progress("phase5", 50),
        ]
        
        await asyncio.gather(*updates)
        
        # verify final state
        status = await run_manager.get_run_status(run_id)
        assert status is not None
        assert 'phase_progress' in status
        assert 'metrics' in status
    
    @pytest.mark.asyncio
    async def test_fifo_ordering(self, run_manager, db_pool):
        """Test that runs are claimed in FIFO order."""
        # create runs with small delays to ensure order
        run_ids = []
        for i in range(5):
            run_id = await run_manager.create_run(
                input_hash=f"test_hash_fifo_{i}",
                options={"order": i},
                schema_version="test"
            )
            run_ids.append(run_id)
            await asyncio.sleep(0.1)  # ensure different timestamps
        
        # claim runs and verify order
        claimed_order = []
        for _ in range(5):
            run_data = await run_manager.claim_run("worker_1")
            if run_data:
                claimed_order.append(run_data['run_id'])
        
        # verify FIFO order
        assert claimed_order == run_ids
    
    @pytest.mark.asyncio
    async def test_completion_prevents_reclaim(self, run_manager):
        """Test that completed runs cannot be reclaimed."""
        # create and claim a run
        run_id = await run_manager.create_run(
            input_hash="test_hash_complete",
            options={"test": True},
            schema_version="test"
        )
        
        await run_manager.claim_run("worker_1")
        
        # complete the run
        await run_manager.complete_run(
            run_id,
            'succeeded',
            {'test': 'metrics'}
        )
        
        # try to claim - should get nothing
        run_data = await run_manager.claim_run("worker_2")
        assert run_data is None
        
        # verify state
        status = await run_manager.get_run_status(run_id)
        assert status['state'] == 'succeeded'


class TestCacheConcurrency:
    """Test concurrent cache operations."""
    
    @pytest.fixture
    async def db_pool(self):
        """Create test database pool."""
        pool = DatabasePool()
        await pool.initialize(
            host=os.getenv('TEST_POSTGRES_HOST', 'localhost'),
            database=os.getenv('TEST_POSTGRES_DB', 'centrifuge_test')
        )
        
        # clean up test mappings
        await pool.execute("""
            DELETE FROM canonical_mappings 
            WHERE model_id = 'test_model'
        """)
        
        yield pool
        
        # cleanup
        await pool.execute("""
            DELETE FROM canonical_mappings 
            WHERE model_id = 'test_model'
        """)
        await pool.close()
    
    @pytest.fixture
    async def cache(self, db_pool):
        """Create cache instance."""
        from core.database import CanonicalMappingCache
        return CanonicalMappingCache(db_pool)
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_writes(self, cache):
        """Test concurrent cache writes don't cause conflicts."""
        # simulate multiple workers writing to cache
        async def store_mapping(worker_id: int):
            """Store mapping in cache."""
            await cache.store_mapping(
                column_name="Department",
                variant_value=f"Dept_{worker_id}",
                canonical_value="IT",
                model_id="test_model",
                confidence=0.9
            )
        
        # run concurrent writes
        await asyncio.gather(*[
            store_mapping(i) for i in range(10)
        ])
        
        # verify all mappings were stored
        for i in range(10):
            result = await cache.get_mapping(
                "Department",
                f"Dept_{i}",
                "test_model"
            )
            assert result == "IT"
    
    @pytest.mark.asyncio
    async def test_cache_superseding(self, cache):
        """Test that newer mappings supersede older ones."""
        # store initial mapping
        await cache.store_mapping(
            column_name="Department",
            variant_value="Tech",
            canonical_value="IT",
            model_id="test_model",
            prompt_version="v1",
            confidence=0.8
        )
        
        # verify initial mapping
        result = await cache.get_mapping("Department", "Tech", "test_model")
        assert result == "IT"
        
        # store updated mapping
        await cache.store_mapping(
            column_name="Department",
            variant_value="Tech",
            canonical_value="Engineering",
            model_id="test_model",
            prompt_version="v2",
            confidence=0.95
        )
        
        # verify new mapping is returned
        result = await cache.get_mapping("Department", "Tech", "test_model")
        assert result == "Engineering"


if __name__ == "__main__":
    # run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])