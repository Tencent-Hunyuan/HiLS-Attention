from utils.landmark_utils import *

import pytest
@pytest.mark.parametrize("ids, chunk_size", [
    ([1,2,3,4,5,6,7], 4),
    ([1,2,3,4,5,6], 4),
    [[i for i in range(8192)], 64]
])
def test_insert_lmk_tokens(ids, chunk_size) -> None:
    lmk_id = -100
    new_ids = [0] * (len(ids) // (chunk_size - 1) + len(ids))
    old_i = 0
    for i in range(len(new_ids)):
        if (i + 1) % chunk_size == 0:
            new_ids[i] = lmk_id
        else:
            new_ids[i] = ids[old_i]
            old_i += 1

    new_ids_torch = torch.tensor(ids).unsqueeze(0)
    new_ids_torch = insert_special_tokens(new_ids_torch, lmk_id, chunk_size)

    assert torch.all(new_ids_torch.squeeze(0) == torch.tensor(new_ids))

@pytest.mark.parametrize("seq_length, chunk_size", [
    (8, 4),
    (8192, 64),
    (8196, 64)
])
def test_create_position_ids_with_landmarks(seq_length, chunk_size) -> None:
    org_pos = [[i for i in range(seq_length)]]
    new_pos = [[0] * (seq_length // (chunk_size - 1) + seq_length)]
    old_i = 0
    for i in range(len(new_pos[0])):
        if (i + 1) % chunk_size == 0:
            new_pos[0][i] = org_pos[0][old_i]
        else:
            new_pos[0][i] = org_pos[0][old_i]
            old_i += 1

    new_pos_torch = create_position_ids_with_landmarks(None, seq_length, chunk_size, torch.device("cpu"))

    assert torch.all(new_pos_torch == torch.tensor(new_pos))