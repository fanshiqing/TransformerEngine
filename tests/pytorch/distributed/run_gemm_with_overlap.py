#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import socket
import warnings
import subprocess
import argparse
import operator
from functools import partial, reduce

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.fp8 import _default_sf_compute

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

nvte_comm_types = {
    "rs": 0,
    "ag": 1,
}


def _mapped_argtype(opt, typemap):
    if str(opt).lower() not in typemap.keys():
        raise TypeError(f"Unrecognized option! Please choose from: {typemap.keys()}")
    return typemap[str(opt).lower()]


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Test comm+GEMM overlap with Userbuffers.")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("-s", "--seq-length", type=int, default=512, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=12, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=64, help="Dimension of each attention head."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--fp8-output", action="store_true", default=False, help="Get FP8 output from GEMM."
    )
    parser.add_argument(
        "--p2p", action="store_true", default=False, help="Test overlap with P2P comms."
    )
    parser.add_argument(
        "--atomic", action="store_true", default=False, help="Test overlap with atomic GEMM."
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        default=False,
        help="Aggregate 2X chunks for P2P split pipelined all-gather.",
    )
    parser.add_argument(
        "--comm-type",
        type=partial(_mapped_argtype, typemap=nvte_comm_types),
        default=0,
        help="Comm type to overlap.",
    )
    parser.add_argument(
        "--bulk-overlap",
        action="store_true",
        default=False,
        help="Enable bulk AG or RS overlap for a tensor that is not involved in the GEMM compute.",
    )
    parser.add_argument(
        "--check-numerics",
        action="store_true",
        default=False,
        help="Test numerical result against torch.matmul(...)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        help="Run some warmup iterations of the comm+GEMM overlap before " + "the timing runs.",
    )
    parser.add_argument(
        "--timing-iters",
        type=int,
        default=1,
        help="Benchmark the comm+GEMM overlap as an average of many iterations.",
    )
    parser.add_argument(
        "--clock-speed",
        type=int,
        default=-1,
        help="Set device clock speed to a fixed value via `nvidia-smi`.",
    )
    parser.add_argument(
        "--std", type=float, default=0.023, help="Standard deviation for input and weight tensors."
    )
    parser.add_argument(
        "--tcp-init",
        action="store_true",
        default=False,
        help="Initialize torch.distributed with TcpStore.",
    )
    parser.add_argument(
        "--init-method", type=str, default=None, help="Set the torch.distributed init method."
    )
    parser.add_argument(
        "--bind-to-device",
        action="store_true",
        default=False,
        help=(
            "Initialize torch.distributed with 'device_id' argument to bind each rank to 1 device."
        ),
    )
    parser.add_argument(
        "--bootstrap-backend",
        type=str.lower,
        default="nccl",
        choices=["gloo", "mpi", "nccl"],
        help=(
            "PyTorch distributed backend for host tensor collectives during comm+GEMM overlap "
            + "initialization."
        ),
    )
    parser.add_argument(
        "--use-cuda-graphs", action="store_true", default=False, help="Use CUDA graphs."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Verbose info messages."
    )
    opts = parser.parse_args(argv, namespace)

    if opts.bulk_overlap:
        if opts.p2p:
            warnings.warn("Point-2-point comms are not supported with bulk overlap.")
            opts.p2p = False
        if opts.atomic:
            warnings.warn("Atomic GEMM is not supported with bulk overlap.")
            opts.atomic = False
        if opts.fp8:
            warnings.warn("Bulk overlap is supported in FP8 but only tested in BF16.")
            opts.fp8 = False
    elif opts.comm_type == 1:
        if opts.atomic:
            setattr(opts, "atomic_rs_p2p", opts.p2p)
        if not opts.p2p:
            warnings.warn("All-gather overlap is only supported with point-2-point comms.")
            opts.p2p = True

    if opts.atomic:
        if not te.fp8.check_fp8_support():
            assert not opts.fp8, "Atomic GEMM is only supported in FP8."
        opts.fp8 = True

    return opts


@record
def _main(opts):
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        # Execution with `mpirun -np N`
        WORLD_RANK = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
        WORLD_SIZE = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1"))
        opts.tcp_init = True
        opts.bootstrap_backend = "mpi"
    elif "TORCHELASTIC_RUN_ID" in os.environ:
        WORLD_RANK = int(os.getenv("RANK", "0"))
        WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
        LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
        LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    else:
        raise RuntimeError(f"{__file__} must be launched with either `mpirun` or `torchrun`!")
    assert WORLD_SIZE == LOCAL_SIZE  # this test supports only 1 node
    assert LOCAL_SIZE <= torch.cuda.device_count()

    assert opts.bulk_overlap == False
    assert opts.fp8 == False
    assert opts.atomic == False

    # Fix clock speed
    torch.cuda.set_device(LOCAL_RANK)
    if opts.clock_speed > 0:
        subprocess.run(
            ["nvidia-smi", "-pm", "ENABLED", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = subprocess.run(
            ["nvidia-smi", "-lgc", str(opts.clock_speed), "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        msg = result.stdout.decode("utf-8").splitlines()[0]
        print(f"[rank:{LOCAL_RANK}] {msg}\n", end="", flush=True)

    # Info printout
    def dist_print(msg, src=None, info=False, error=False, section=False, group=None):
        group = dist.new_group() if group is None else group
        rank = dist.get_rank(group)
        stream = sys.stderr if error else sys.stdout
        if info or opts.verbose:
            if section:
                if rank == (0 if src is None else src):
                    stream.write("\n")
                dist.barrier(group)
            if src is None or rank == src:
                prefix = "[GLOBAL] " if src is not None else f"[rank:{rank}] "
                lines = msg.splitlines()
                msg = "\n".join(
                    [prefix + lines[0]] + [(" " * len(prefix)) + line for line in lines[1:]]
                )
                stream.write(msg + "\n")
            dist.barrier(group)

    # Initialize torch.distributed global process group and get TP group
    dist_init_kwargs = {
        "backend": "nccl",
        "rank": WORLD_RANK,
        "world_size": WORLD_SIZE,
    }
    if opts.tcp_init:
        if opts.init_method is not None:
            assert opts.init_method.startswith("tcp://")
            init_method = opts.init_method
        else:
            MASTER_ADDR = os.getenv("MASTER_ADDR", socket.gethostbyname(socket.gethostname()))
            MASTER_PORT = os.getenv("MASTER_PORT", "1234")
            init_method = f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
        dist_init_kwargs["init_method"] = init_method
    elif opts.init_method is not None:
        assert (
            opts.init_method.startswith("env://")
            or opts.init_method.startswith("file://")
            or opts.init_method.startswith("tcp://")
        )
        dist_init_kwargs["init_method"] = opts.init_method
    if opts.bind_to_device or opts.bootstrap_backend == "nccl":
        dist_init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")
    assert dist.is_nccl_available()
    dist.init_process_group(**dist_init_kwargs)
    tp_group = dist.new_group(backend="nccl")
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    dist_print(
        f"Initialized default NCCL process group with {tp_size} GPUs",
        src=0,
        section=True,
        info=True,
        group=tp_group,
    )

    # Initialize backend used in bootstrapping Userbuffers
    if opts.bootstrap_backend == "gloo":
        assert dist.is_gloo_available()
    elif opts.bootstrap_backend == "mpi":
        assert dist.is_mpi_available()
    bootstrap_pg = dist.new_group(backend=opts.bootstrap_backend)
    dist_print(
        f'Bootstrapping comm+GEMM overlap with backend="{opts.bootstrap_backend}"',
        src=0,
        section=True,
        info=True,
        group=bootstrap_pg,
    )
    if WORLD_RANK == 0:
        print("\n", end="", flush=True)

    ub_callbacks = (
        tex.UbufBootstrapCallbacks()
        if tex.ubuf_built_with_mpi()
        else tex.UbufBootstrapCallbacks(bootstrap_pg, bootstrap_pg)
    )

    if opts.comm_type == 0:
        if opts.bulk_overlap:
            ub_algo = tex.UbufOverlapAlgo.BULK_OVERLAP_RS
        elif opts.p2p:
            ub_algo = (
                tex.UbufOverlapAlgo.ATOMIC_GEMM_RS_P2P
                if opts.atomic
                else tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS_P2P
            )
        else:
            ub_algo = (
                tex.UbufOverlapAlgo.ATOMIC_GEMM_RS
                if opts.atomic
                else tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS
            )
    elif opts.comm_type == 1:
        if opts.bulk_overlap:
            ub_algo = tex.UbufOverlapAlgo.BULK_OVERLAP_AG
        else:
            ub_algo = (
                tex.UbufOverlapAlgo.ATOMIC_GEMM_AG_P2P
                if opts.atomic
                else tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P
            )
    else:
        raise TypeError("Invalid comm+GEMM overlap type!")

    # Initialize userbuffers with (M, N) buffer
    # M = sequence * batch
    # N = hidden size
    hidden_size = opts.num_heads * opts.head_dim
    inp_shape = (opts.seq_length, opts.batch_size, hidden_size)
    outer_size = reduce(operator.mul, inp_shape[:-1], 1)
    ubuf_dtype = torch.bfloat16
    sample_buffer = torch.empty((outer_size, hidden_size), dtype=ubuf_dtype, device="cuda")
    ub_obj = ub_obj = (
        tex.UbufP2PCommOverlap(
            sample_buffer,  # Sample userbuffer
            WORLD_RANK,  # World rank
            WORLD_SIZE,  # World size
            LOCAL_RANK,  # Rank within the node
            LOCAL_SIZE,  # Number of ranks/GPUs per node
            0,  # Node ID
            1,  # Number of nodes
            tp_size,  # Tensor-parallel group size (may be different than LOCAL_SIZE)
            1,  # Number of communication SMs
            1,  # CGA cluster size
            opts.comm_type == 0 or opts.atomic,  # Set SM margin
            opts.aggregate,  # Aggregate 2X GEMM chunks
            3,  # Max concurrent GEMM streams
            opts.comm_type == 0,  # overlap with reduce scatter
            opts.atomic,  # use a single GEMM with atomic-counters
            not (opts.atomic and bool(int(os.getenv("NVTE_AG_P2P_MULTI_ATOMIC", "0")))),
            ub_callbacks,
        )
        if opts.p2p
        else tex.UbufCommOverlap(
            sample_buffer,  # Sample userbuffer
            WORLD_RANK,  # World rank
            WORLD_SIZE,  # World size
            LOCAL_RANK,  # Rank within the node
            LOCAL_SIZE,  # Number of ranks/GPUs per node
            0,  # Node ID
            1,  # Number of nodes
            tp_size,  # Tensor-parallel group size (may be different than LOCAL_SIZE)
            16,  # Number of communication SMs
            2,  # CGA cluster size
            4,  # Number of communication splits
            True,  # Set SM margin
            3,  # Max concurrent GEMM streams
            opts.atomic,  # Use a single GEMM with atomic-counters
            ub_callbacks,
        )
    )

    # Figure out problem sizing:
    # M = sequence * batch
    # N = hidden size
    # K = MLP intermediate size (usually 4x hidden size)
    # P = number of devices for sequence/tensor parallelism
    # NOTE: TE-GEMM is set up to work with a transposed kernels and  non-transposed inputs.
    # ffn_hidden_size = 4 * hidden_size
    ffn_hidden_size = 11008
    if opts.comm_type == 1:
        # (M/P, N) -> overlapped AG -> (M, N) x (K/P, N)^T = (M, K/P)
        local_kernel_t_shape = (ffn_hidden_size // tp_size, hidden_size)
        local_inp_shape = (outer_size // tp_size, hidden_size)
    else:
        # (M, K/P) x (N, K/P)^T = (M, N) -> overlapped RS -> (M/P, N)
        local_kernel_t_shape = (hidden_size, ffn_hidden_size // tp_size)
        local_inp_shape = (outer_size, ffn_hidden_size // tp_size)

    # Initialize distributed input tensor and GEMM kernels
    torch.manual_seed(opts.seed + tp_rank)
    torch.cuda.manual_seed(opts.seed + tp_rank)
    inp = torch.nn.init.normal_(
        torch.empty(local_inp_shape, dtype=torch.bfloat16, device="cuda"),
        mean=0.0,
        std=opts.std,
    )
    kernel_t = torch.nn.init.normal_(
        torch.empty(local_kernel_t_shape, dtype=torch.bfloat16, device="cuda"),
        mean=0.0,
        std=opts.std,
    )

    # Gather global tensors and calculate reference result (need these first for Fp8 scales)
    if opts.comm_type == 1:
        # AG Kernel: (K/P, N) -> gather -> (K, N) -> T -> (N, K)
        ker_g = torch.transpose(
            te.distributed.gather_along_first_dim(kernel_t, tp_group)[0], 0, 1
        ).to(dtype=torch.float32)
        # AG Input: (M/P, N) -> gather -> (M, N)
        inp_g = te.distributed.gather_along_first_dim(inp, tp_group)[0].to(dtype=torch.float32)
    else:
        # RS Kernel: (N, K/P) -> T -> (K/P, N) -> gather -> (K, N)
        ker_g = te.distributed.gather_along_first_dim(
            torch.transpose(kernel_t, 0, 1), tp_group
        )[0].to(dtype=torch.float32)
        # RS Input: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
        inp_g = torch.transpose(
            te.distributed.gather_along_first_dim(torch.transpose(inp, 0, 1), tp_group)[0], 0, 1
        ).to(dtype=torch.float32)

  
    ref_g = torch.matmul(inp_g, ker_g)

    # Set up comm/compute buffers
    if opts.comm_type == 1:
        ub_obj.copy_input_to_ubuf(inp, 1)
        gemm_inp = ub_obj.get_ubuf_output(1)
        ubuf_out = None
        rs_out = None
    else:
        ubuf_out = ub_obj.get_ubuf_output(1)
        gemm_inp = inp
        rs_out = torch.empty(
            (outer_size // tp_size, hidden_size), dtype=torch.bfloat16, device="cuda"
        )

    def _gemm():
        import pdb
        # pdb.set_trace()
        return tex.gemm(
            kernel_t,
            gemm_inp,
            torch.bfloat16,
            te.module.base.get_workspace(),
            bias=None,
            use_bias=False,
            # grad=True,
            # layout="NN",
            gelu=False,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=rs_out,
            out=ubuf_out,
        )

    # Trigger GEMM
    total_iters = opts.warmup_iters + opts.timing_iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    torch.cuda.synchronize()

    for i in range(total_iters):
        start_events[i].record()
        all_outputs = _gemm()
        end_events[i].record()

    torch.cuda.synchronize()
    gpu_times = [
        s.elapsed_time(e)
        for s, e in zip(start_events[opts.warmup_iters :], end_events[opts.warmup_iters :])
    ]

    avg_gpu_time = sum(gpu_times) / opts.timing_iters
    gemm_name = "".join(
        [
            "p2p all-gather + " if opts.comm_type == 1 else "",
            "atomic " if opts.atomic else "",
            "GEMM",
            (f" + {'p2p ' if opts.p2p else ''}reduce-scatter" if opts.comm_type == 0 else ""),
        ]
    )
    timing_info = (
        f"Avg. GPU time for {gemm_name}: {avg_gpu_time} ms "
        + f"({opts.warmup_iters} warmup + {opts.timing_iters} timing runs)"
    )
    dist_print(timing_info, section=True, info=True, group=tp_group)

    # Compare against standard GEMM
    numerics_failed = False
    if opts.check_numerics:
        torch.cuda.synchronize()
        dist.barrier(tp_group)
        if opts.comm_type == 1:
            # AG Output: (M, K/P) -> T -> (K/P, M) -> gather -> (K, M) -> T -> (M, K)
            output = (
                all_outputs[0]
            )
            test_out = torch.transpose(
                te.distributed.gather_along_first_dim(
                    torch.transpose(output, 0, 1), tp_group
                )[0],
                0,
                1,
            )
        else:
            # RS Output: (M/P, N) -> gather -> (M, N)
            output = rs_out.to(dtype=torch.float32)
            test_out = te.distributed.gather_along_first_dim(output, tp_group)[0]


        ref_out = ref_g
        test_nonzeros = torch.count_nonzero(test_out)
        ref_nonzeros = torch.count_nonzero(ref_out)
        nonzero_info = (
            f"output nonzeros = {test_nonzeros} " + f"| reference count = {ref_nonzeros}"
        )
        dist_print(nonzero_info, src=0, section=True, group=tp_group)

        sizing_info = (
            f"input: {list(inp.shape)} " + f"| GEMM1 weights: {list(kernel_t.shape)[::-1]} "
        )
        sizing_info += f"| output: {list(output.shape)}\n"
        dist_print(sizing_info, section=True, group=tp_group)

        sizing_info_g = (
            f"input: {list(inp_g.shape)} " + f"| GEMM1 weights: {list(ker_g.shape)} "
        )
        sizing_info_g += (
            f"| output: {list(test_out.shape)} " + f"| reference: {list(ref_out.shape)}\n"
        )
        dist_print(sizing_info_g, src=0, group=tp_group)

        torch.cuda.synchronize()
        dist.barrier(tp_group)
        diff = torch.abs(test_out - ref_out).flatten()
        m = torch.argmax(diff)
        abs_err = diff[m].item()
        rel_err = abs_err / max(abs(ref_out.flatten()[m].item()), 1e-5)
        rtol = 0.125 if opts.fp8 else 0.02
        atol = 0.0625 if opts.fp8 else 0.001
        if rel_err > rtol and abs_err > atol:
            numerics_failed = True
            numerics_info = (
                "NUMERICAL CHECK FAILED: "
                + f"Outputs not close enough at index {m.item()} "
                + f"with {test_out.flatten()[m].item()} vs {ref_out.flatten()[m].item()} | "
                + f"rel. error = {rel_err} (tol = {rtol}) | "
                + f"abs. error = {abs_err} (tol = {atol})"
            )
        else:
            numerics_info = "NUMERICAL CHECK PASSED: "
            if rel_err <= rtol:
                numerics_info += f"rel. error = {rel_err} (tol = {rtol})" + (
                    " | " if abs_err < atol else ""
                )
            if abs_err <= atol:
                numerics_info += f"abs. error = {abs_err} (tol = {atol})"

        dist_print(
            numerics_info, src=0, section=True, info=True, error=numerics_failed, group=tp_group
        )

    dist.barrier(tp_group)
    if LOCAL_RANK == 0:
        print("\n", end="", flush=True)

    dist.destroy_process_group()

    # Reset clock speeds
    if opts.clock_speed > 0:
        subprocess.run(
            ["nvidia-smi", "-pm", "ENABLED", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = subprocess.run(
            ["nvidia-smi", "-rgc", "-i", str(LOCAL_RANK)],
            env=os.environ,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return int(numerics_failed)


if __name__ == "__main__":
    sys.exit(_main(_parse_args()))
