import os
from transformers import AutoTokenizer
import numpy as np
import subprocess
from glob import glob
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import json
import shutil
from tqdm import tqdm
import pandas as pd
import argparse
import io
import sys


def _tokenize_jsonl(path, tokenizer, output_dir):
    texts = []
    lens = []

    filename = os.path.basename(path)
    base_name = filename 
    output_path = os.path.join(output_dir, base_name)
    output_path = output_path + ".data"

    
    if os.path.exists(output_path):
        fsize = os.path.getsize(output_path)
        if fsize > 0:
            print(f"File {output_path} already exists. Skipping...")
            return
    print(f'Tokenizing file: {path} -> {output_path}')

    with open(path, 'r', encoding='utf-8') as fh:
        accum_conext = ''  # 初始化累积缓冲区
        for line in fh:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                accum_conext += line
                obj = json.loads(accum_conext)
                accum_conext = ''
                encoded = tokenizer.encode(obj['text'])
                texts.extend(encoded)
                texts.append(tokenizer.eos_token_id)
            except json.JSONDecodeError as e:
                print(line)
                continue

    tokens = np.array(texts, dtype=np.uint32)
    tokens.tofile(output_path)
    out_size = os.path.getsize(output_path)
    print(f"tokenized: {output_path}, size: {out_size}")
        # with open(output_path + ".len.pkl", 'wb') as f:
        #     pickle.dump(lens, f)
    return
    

def process_file(file_path, input_root, output_root, tokenizer):
    """处理单个文件的原子操作"""
    # 构建相对路径
    rel_path = os.path.relpath(file_path, start=input_root)
    output_dir = os.path.join(output_root, os.path.dirname(rel_path))
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        _tokenize_jsonl(file_path, tokenizer, output_dir)
    except Exception as e:
        print(f"\nError processing {os.path.basename(file_path)}: {str(e)}")
        raise

def file_generator(input_path):
    """
    生成器函数：惰性遍历目录下的所有文件
    避免一次性加载所有文件路径到内存
    """
    for root, dirs, files in os.walk(input_path):
        for file in files:
            yield os.path.join(root, file)


def concurrent_tokenize(input_path, output_path, tokenizer, max_workers=None):
    """
    生产者-消费者模式的并发处理
    - 每次最多保持 max_workers 个任务在运行
    - 每完成一个任务，立即从生成器获取下一个文件并提交
    """
    os.makedirs(output_path, exist_ok=True)
    
    workers = max_workers or multiprocessing.cpu_count()
    print(f"Processing {input_path} with Producer-Consumer pattern, {workers} workers...")
    
    # 创建文件生成器（惰性加载）
    file_iter = file_generator(input_path)
    files_exhausted = False
    
    # 统计计数
    submitted_count = 0
    processed_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # futures 字典: future -> file_path (用于追踪)
        futures = {}
        
        # ============ 初始化：填满工作队列 ============
        for _ in range(workers):
            try:
                file_path = next(file_iter)
                future = executor.submit(
                    process_file,
                    file_path,
                    input_path,
                    output_path,
                    tokenizer
                )
                futures[future] = file_path
                submitted_count += 1
            except StopIteration:
                files_exhausted = True
                break
        
        print(f"Initial batch submitted: {submitted_count} files")
        
        # ============ 主循环：处理完成的任务，提交新任务 ============
        while futures:
            # 等待至少一个任务完成
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            
            for future in done:
                file_path = futures.pop(future)
                processed_count += 1
                
                # 获取结果，处理异常
                try:
                    future.result()
                except Exception as e:
                    error_count += 1
                    print(f"[Error {error_count}] Failed: {file_path}: {str(e)}")
                
                # ============ 生产者：提交新任务 ============
                if not files_exhausted:
                    try:
                        new_file_path = next(file_iter)
                        new_future = executor.submit(
                            process_file,
                            new_file_path,
                            input_path,
                            output_path,
                            tokenizer
                        )
                        futures[new_future] = new_file_path
                        submitted_count += 1
                        
                        # 每100个文件打印一次进度
                        if submitted_count % 100 == 0:
                            print(f"Progress: submitted={submitted_count}, "
                                  f"processed={processed_count}, "
                                  f"active={len(futures)}, errors={error_count}")
                                  
                    except StopIteration:
                        files_exhausted = True
                        print(f"All files submitted. Total: {submitted_count}")
    
    # 打印最终统计
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"  Total submitted: {submitted_count}")
    print(f"  Total processed: {processed_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*50}")


def concat_data_files_system(input_dir, output_file):
    chunk_dirs = []
    for root, dirs, _ in os.walk(input_dir):
        dirs[:] = natsorted(dirs, key=lambda x: x.lower())
        for d in dirs:
            if d.startswith("chunk"):
                chunk_dirs.append(os.path.join(root, d))

    chunk_dirs = natsorted(chunk_dirs, key=lambda x: x.lower())
    print(chunk_dirs)

    data_files = []
    for chunk in chunk_dirs:
        files = natsorted(
            glob(os.path.join(chunk, "*.data")),
            key=lambda x: os.path.basename(x).lower()
        )
        data_files.extend(files)

    if not data_files:
        raise FileNotFoundError(f"No .data files found in {input_dir}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    open(output_file, 'wb').close()

    success_count = 0
    try:
        with open(output_file, 'ab') as f_out:
            # for idx, data_file in enumerate(data_files, 1):
            for idx, data_file in enumerate(tqdm(data_files, desc="Merging files", unit="file", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'), 1):
                # print(f"Merging [{idx}/{len(data_files)}] {os.path.basename(data_file)}")
                with open(data_file, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out, 8 * 1024 * 1024)
                success_count += 1

        print(f"Success: Merged {success_count}/{len(data_files)} files -> {output_file}")

    except Exception as e:
        current_file = data_files[idx-1] if idx else "unknown"
        raise RuntimeError(
            f"Merge failed at file {current_file}: {e}"
        ) from e
    
if __name__ == "__main__":
    cmd = argparse.ArgumentParser('Tokenize Olmo3 datasets')
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--corpus_dir', required=True, type=str)
    cmd.add_argument('--output_dir', required=True, type=str)
    cmd.add_argument('--num_workers', required=True, type=int, default=20)

    args = cmd.parse_args(sys.argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    concurrent_tokenize(
        input_path=args.corpus_dir,
        output_path=args.output_dir,
        tokenizer=tokenizer,
        max_workers=args.num_workers
    )
