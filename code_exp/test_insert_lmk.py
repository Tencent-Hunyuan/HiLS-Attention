import torch


def _insert_special_tokens(input_ids, fill_id, chunk_size):
    N = input_ids.shape[0]
    input_ids_ = input_ids.view(N, -1, chunk_size - 1)  # (N, L / cz, cz)
    chunk_num = input_ids_.shape[1]
    chunk_id_padding = torch.ones(N, chunk_num, 1, device=input_ids.device, dtype=torch.long).fill_(fill_id)
    # chunked_input_ids = torch.cat([input_ids_, chunk_id_padding], dim=2)  # (N, L / cz, cz+1)
    chunked_input_ids = torch.cat([input_ids_, chunk_id_padding], dim=2)  # (N, L / cz, cz+1)
    chunked_input_ids = chunked_input_ids.view(N, -1)  # (N, L // cz * (cz + 1))
    return chunked_input_ids


if __name__ == "__main__":
    input_ids = torch.tensor([[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16]])
    labels = torch.tensor([[1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16]])
    lmk_id = -100
    input_ids = _insert_special_tokens(input_ids, lmk_id, 5)
    labels = torch.roll(labels, shifts=-1, dims=-1)
    labels[:, -1] = -100
    labels = _insert_special_tokens(labels, -100, 5)
    labels = torch.roll(labels, shifts=1, dims=-1)

    print("input_ids:", input_ids)
    print("labels:", labels)
