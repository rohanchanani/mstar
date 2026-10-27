"""Partition registration must preserve resident sampler state."""
from types import SimpleNamespace

import torch

from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource


def _resource(*, planned=False):
    resource = SamplerResource(
        vocab_size=16, enable_repetion_penalty=True,
        device=torch.device("cpu"), planned=planned,
    )
    policy = SamplingReqConfig(temperature=0.9, top_k=8, repetition_penalty=1.05)
    policy.apply_conductor_config(seed=17, resource_key="talker_sampler")
    return resource, policy


def test_duplicate_ingestion_preserves_planned_sampling_history():
    resource, policy = _resource(planned=True)
    resource.ingest_request(7, policy)
    mask = resource._sampler.get_token_mask(7)
    mask.add_tokens(torch.tensor([2, 9]))
    resource._sampler._step_offset[7] = 11
    config = resource._sampler._sampling_config[7]

    resource.ingest_request(7, policy)

    assert resource._sampler._sampling_config[7] is config
    assert resource._sampler.get_token_mask(7) is mask
    assert resource._sampler._step_offset[7] == 11
    assert mask._seen_token_mask.nonzero().flatten().tolist() == [2, 9]


def test_duplicate_ingestion_preserves_graph_slot_and_next_gather():
    resource, policy = _resource()
    resource.build_accelerator_graph_buffers([], 1, 16)
    resource.ingest_request(7, policy)
    buffers = resource._cg_buffers
    buffers.gather_static([7], 1, 0)
    context = SimpleNamespace(request_ids=[7])
    lease = SimpleNamespace(slot=0, bucket=SimpleNamespace(bs=1))
    resource._gather_dynamic(context, lease)
    slot = buffers._rid_to_slot[7]
    mask = resource._sampler.get_token_mask(7)
    mask.add_tokens(torch.tensor([2, 9]))
    resource._sampler._step_offset[7] = 11
    buffers.offset.master[slot] = 11

    resource.ingest_request(7, policy)
    buffers.gather_static([7], 1, 0)
    resource._gather_dynamic(context, lease)

    assert resource._sampler.get_token_mask(7) is mask
    assert resource._sampler._step_offset[7] == 11
    assert buffers._rid_to_slot[7] == slot
    assert int(buffers.offset.master[slot]) == 11
    assert buffers.slice_for_bs(1, 0)["seen_tokens_buf"][0].nonzero().flatten().tolist() == [2, 9]


def test_remove_then_ingest_initializes_a_fresh_request():
    resource, policy = _resource()
    resource.build_accelerator_graph_buffers([], 1, 16)
    resource.ingest_request(7, policy)
    old_mask = resource._sampler.get_token_mask(7)
    old_mask.add_tokens(torch.tensor([2, 9]))
    resource._sampler._step_offset[7] = 11

    resource.remove_request(7)
    resource.ingest_request(7, policy)
    buffers = resource._cg_buffers
    buffers.gather_static([7], 1, 0)
    resource._gather_dynamic(
        SimpleNamespace(request_ids=[7]),
        SimpleNamespace(slot=0, bucket=SimpleNamespace(bs=1)),
    )

    assert resource._sampler.get_token_mask(7) is not old_mask
    assert resource._sampler._step_offset[7] == 0
    assert int(buffers.offset.master[buffers._rid_to_slot[7]]) == 0
    assert not buffers.slice_for_bs(1, 0)["seen_tokens_buf"].any()
