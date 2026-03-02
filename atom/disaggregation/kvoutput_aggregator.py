# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVConnectorOutput:
    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
    
    def is_empty(self) -> bool:
        """Check if both finished_sending and finished_recving are empty."""
        return not self.finished_sending and not self.finished_recving


class KVOutputAggregator:
    """Utility class to aggregate the KVConnectorOutput of all workers into a single
    output corresponding to Rank 0 for scheduler."""

    def __init__(self, world_size: int = 8):
        # Track remaining workers for each request for sending and receiving separately
        self._remaining_sending_workers = dict[str, int]()  # req_id -> remaining_workers_count
        self._remaining_recving_workers = dict[str, int]()  # req_id -> remaining_workers_count
        self._world_size = world_size

    def aggregate(self, worker_outputs: list[KVConnectorOutput]) -> KVConnectorOutput:
        """
        Aggregate KVConnectorOutput from all workers and return completed req_ids.
        
        Args:
            worker_outputs: List of KVConnectorOutput from each worker
            
        Returns:
            KVConnectorOutput containing req_ids that have been completed by ALL workers
        """
        if not worker_outputs:
            return KVConnectorOutput()
        
        # Collect all unique req_ids from all workers for both sending and receiving
        all_sending_req_ids = set()
        all_recving_req_ids = set()
        
        for worker_output in worker_outputs:
            if worker_output.finished_sending:
                all_sending_req_ids.update(worker_output.finished_sending)
            if worker_output.finished_recving:
                all_recving_req_ids.update(worker_output.finished_recving)
        
        # Initialize remaining workers count for new req_ids
        for req_id in all_sending_req_ids:
            if req_id not in self._remaining_sending_workers:
                self._remaining_sending_workers[req_id] = self._world_size
        
        for req_id in all_recving_req_ids:
            if req_id not in self._remaining_recving_workers:
                self._remaining_recving_workers[req_id] = self._world_size
        
        # Update remaining count for each worker's completed req_ids
        for worker_output in worker_outputs:
            # Process sending req_ids
            if worker_output.finished_sending:
                for req_id in worker_output.finished_sending:
                    if req_id in self._remaining_sending_workers:
                        self._remaining_sending_workers[req_id] -= 1
            
            # Process receiving req_ids
            if worker_output.finished_recving:
                for req_id in worker_output.finished_recving:
                    if req_id in self._remaining_recving_workers:
                        self._remaining_recving_workers[req_id] -= 1
        
        # Find req_ids that have reached 0 remaining workers
        finished_sending_req_ids = set()
        finished_recving_req_ids = set()
        
        # Check sending req_ids
        sending_req_ids_to_remove = []
        for req_id, remaining in self._remaining_sending_workers.items():
            if remaining <= 0:
                finished_sending_req_ids.add(req_id)
                sending_req_ids_to_remove.append(req_id)
        
        # Check receiving req_ids
        recving_req_ids_to_remove = []
        for req_id, remaining in self._remaining_recving_workers.items():
            if remaining <= 0:
                finished_recving_req_ids.add(req_id)
                recving_req_ids_to_remove.append(req_id)
        
        # Clean up finished req_ids
        for req_id in sending_req_ids_to_remove:
            del self._remaining_sending_workers[req_id]
        
        for req_id in recving_req_ids_to_remove:
            del self._remaining_recving_workers[req_id]
        
        return KVConnectorOutput(
            finished_sending=finished_sending_req_ids,
            finished_recving=finished_recving_req_ids
        )


# Test code
if __name__ == "__main__":
    print("Test 1: Basic scenario")
    aggregator = KVOutputAggregator(world_size=8)
    
    # Round 1: All workers output empty sets
    worker_outputs = [KVConnectorOutput() for _ in range(8)]
    finished = aggregator.aggregate(worker_outputs)
    print(f"Round 1 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    
    # Round 2: All workers finish sending req_id="1"
    worker_outputs = [KVConnectorOutput(finished_sending={"1"}) for _ in range(8)]
    finished = aggregator.aggregate(worker_outputs)
    print(f"Round 2 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    
    # Round 3: All workers finish receiving req_id="1"
    worker_outputs = [KVConnectorOutput(finished_recving={"1"}) for _ in range(8)]
    finished = aggregator.aggregate(worker_outputs)
    print(f"Round 3 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    
    print("\nTest 2: Complex scenario")
    aggregator2 = KVOutputAggregator(world_size=8)
    
    # Round 1: Partial workers complete partial requests
    worker_outputs = [
        KVConnectorOutput(finished_sending={"1", "2"}, finished_recving={"1"}),
        KVConnectorOutput(finished_sending={"1"}),
        KVConnectorOutput(finished_sending={"2"}, finished_recving={"2"}),
        KVConnectorOutput(finished_sending={"1"}),
        KVConnectorOutput(),  # Empty output
        KVConnectorOutput(finished_sending={"1"}, finished_recving={"3"}),
        KVConnectorOutput(finished_sending={"2"}),
        KVConnectorOutput(finished_sending={"1"})
    ]
    finished = aggregator2.aggregate(worker_outputs)
    print(f"Round 1 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    
    # Round 2: Remaining workers complete
    worker_outputs = [
        KVConnectorOutput(finished_sending={"3"}, finished_recving={"3"}),
        KVConnectorOutput(finished_sending={"2", "3"}, finished_recving={"1"}),
        KVConnectorOutput(finished_sending={"3"}, finished_recving={"2"}),
        KVConnectorOutput(finished_sending={"2"}, finished_recving={"1"}),
        KVConnectorOutput(finished_sending={"1", "2"}, finished_recving={"2"}),
        KVConnectorOutput(finished_sending={"2", "3"}, finished_recving={"1"}),
        KVConnectorOutput(finished_sending={"1"}, finished_recving={"2"}),
        KVConnectorOutput(finished_sending={"2"}, finished_recving={"3"})
    ]
    finished = aggregator2.aggregate(worker_outputs)
    print(f"Round 2 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    
    print("\nTest 3: Gradual completion scenario")
    aggregator3 = KVOutputAggregator(world_size=3)  # Use smaller world_size
    
    # Round 1: Only 2 workers finish sending
    worker_outputs = [
        KVConnectorOutput(finished_sending={"A"}),
        KVConnectorOutput(finished_sending={"A"}),
        KVConnectorOutput()
    ]
    finished = aggregator3.aggregate(worker_outputs)
    print(f"Round 1 finished_sending: {finished.finished_sending}")
    
    # Round 2: Last worker finishes sending
    worker_outputs = [
        KVConnectorOutput(finished_sending={"B"}),
        KVConnectorOutput(finished_recving={"A"}),
        KVConnectorOutput(finished_sending={"A"})  # Last worker finishes sending
    ]
    finished = aggregator3.aggregate(worker_outputs)
    print(f"Round 2 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")