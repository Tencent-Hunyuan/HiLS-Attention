from turtle import position
import torch


def insert_special_tokens(input_ids, fill_id, chunk_size):
    N, L = input_ids.shape
    full_chunks = L // (chunk_size - 1)
    remainder = L % (chunk_size - 1)
    
    parts = []
    if full_chunks > 0:
        chunk_part = input_ids[:, :full_chunks * (chunk_size - 1)].view(N, full_chunks, chunk_size - 1)
        fill_tokens = torch.full((N, full_chunks, 1), fill_id, device=input_ids.device, dtype=input_ids.dtype)
        parts.append(torch.cat([chunk_part, fill_tokens], dim=2).view(N, -1))
    
    if remainder > 0:
        parts.append(input_ids[:, full_chunks * (chunk_size - 1):])
    
    return torch.cat(parts, dim=1)


def create_position_ids_with_landmarks(position_ids, seq_length, chunk_size, device):
    # args: position_ids: [B, L] or None
    L = seq_length
    if position_ids is None:
        position_ids = torch.arange(0, seq_length, device=device).unsqueeze(0)

    full_chunks = L // (chunk_size - 1)
    remainder = L % (chunk_size - 1)

    result_parts = []

    if full_chunks > 0:
        B = position_ids.shape[0]
        full_part = position_ids[:, :full_chunks * (chunk_size - 1)]

        # repeat lmk pos
        full_part = full_part.view(B, -1, chunk_size - 1)
        last_pos = full_part[:, :, -1:] + 1 
        full_part = torch.cat([full_part, last_pos], dim=-1) 
        full_part = full_part.view(B, -1)
        result_parts.append(full_part)
    
    if remainder > 0:
        remainder_part = position_ids[:, full_chunks * (chunk_size - 1):]  # (N, remainder)
        result_parts.append(remainder_part)
    
    pos = torch.cat(result_parts, dim=-1)
    return pos
