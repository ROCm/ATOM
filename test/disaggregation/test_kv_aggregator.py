# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from atom.disaggregation.kvoutput_aggregator import KVConnectorOutput, KVOutputAggregator
def test_kv_aggregator():
    print("Test 1: Basic scenario")
    aggregator = KVOutputAggregator(world_size=8)
    
    # Phase 1: All workers output empty sets
    worker_outputs = [KVConnectorOutput() for _ in range(8)]
    finished = aggregator.aggregate(worker_outputs)
    print(f"Round 1 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    assert finished.finished_sending == set()
    assert finished.finished_recving == set()
    
    # Phase 2: All workers have finished sending req_id="1"
    worker_outputs = [
        KVConnectorOutput(finished_sending={"1"}) for _ in range(8)
    ]
    finished = aggregator.aggregate(worker_outputs)
    print(f"Round 2 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    assert finished.finished_sending == {"1"}
    assert finished.finished_recving == set()
    
    # Phase 3: All workers have finished receiving req_id="1"
    worker_outputs = [
        KVConnectorOutput(finished_recving={"1"}) for _ in range(8)
    ]
    finished = aggregator.aggregate(worker_outputs)
    print(f"Round 3 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    assert finished.finished_sending == set()
    assert finished.finished_recving == {"1"}
    
    print("\nTest 2: Complex scenario")
    aggregator2 = KVOutputAggregator(world_size=8)
    
    # First round: Some workers complete some requests
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
    assert finished.finished_sending == set()  # No request completed by all workers
    assert finished.finished_recving == set()  # No request completed by all workers
    
    # Second round: Remaining workers complete
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
    assert finished.finished_sending == {"2"}  # All three requests now completed by all workers
    assert finished.finished_recving == set()  # All three requests now completed by all workers
    
    print("\nTest 3: Progressive completion scenario")
    aggregator3 = KVOutputAggregator(world_size=3)  # Using smaller world_size
    
    # Round 1: Only 2 workers finished sending
    worker_outputs = [
        KVConnectorOutput(finished_sending={"1"}),
        KVConnectorOutput(finished_sending={"1"}),
        KVConnectorOutput()
    ]
    finished = aggregator3.aggregate(worker_outputs)
    print(f"Round 1 finished_sending: {finished.finished_sending}")
    assert finished.finished_sending == set()  # Not all workers completed sending
    assert finished.finished_recving == set()
    
    # Round 2: Last worker also finished sending
    worker_outputs = [
        KVConnectorOutput(finished_sending={"2"}),
        KVConnectorOutput(finished_recving={"1"}),
        KVConnectorOutput(finished_sending={"1"})  # Last worker finished sending
    ]
    finished = aggregator3.aggregate(worker_outputs)
    print(f"Round 2 finished_sending: {finished.finished_sending}, finished_recving: {finished.finished_recving}")
    assert finished.finished_sending == {"1"}  # All workers now completed sending "A"
    assert finished.finished_recving == set()  # Not all workers completed receiving

if __name__ == "__main__":
    test_kv_aggregator()
    print("\nAll tests passed!")