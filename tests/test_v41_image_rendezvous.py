from atom.plugin.vllm.v41_state_image import _StepRendezvous
def test_either_order_completes():
    for first, second in (("slots","blocks"), ("blocks","slots")):
        r=_StepRendezvous()
        assert r.offer(1, **{first:[1,2]}) is None
        rows=r.offer(1, **{second:[7,8]})
        assert rows is not None and len(rows)==2
def test_a_half_step_is_counted_not_silent():
    r=_StepRendezvous()
    r.offer(1, slots=[1]); r.offer(2, slots=[1]); 
    assert r.half_steps==1
def test_a_completed_pair_does_not_fire_twice():
    r=_StepRendezvous()
    r.offer(1, slots=[1]); assert r.offer(1, blocks=[5]) is not None
    assert r.offer(1, blocks=[5]) is None
