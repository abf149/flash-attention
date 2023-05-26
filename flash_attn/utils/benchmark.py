# Copyright (c) 2022, Tri Dao.
""" Useful functions for writing test code. """

import torch
import torch.utils.benchmark as benchmark


def benchmark_forward(fn, *inputs, repeats=10, desc='', verbose=True, amp=False,
                      amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward pass of an arbitrary function. """

    #print("Checking flash_attn_unpadded_qkvpacked_func...")
    #print(fn.__name__)
    #if not('flash_attn_unpadded_qkvpacked_func' in fn.__name__):
    #    print("Popping...")
    #    kwinputs.pop('return_attn_probs')

    if verbose:
        print(desc, '- Forward pass')
    def fn_amp(*inputs, **kwinputs):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    for _ in range(repeats):  # warmup
        fn_amp(*inputs, **kwinputs)
    t = benchmark.Timer(
            stmt='fn_amp(*inputs, **kwinputs)',
            globals={'fn_amp': fn_amp, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, amp=False,
                       amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the backward pass of an arbitrary function. """

    y=[]
    sm_lse=[]

    if verbose:
        print(desc, '- Backward pass')
    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):

        if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:
            # CUDA implementation supports LSE backprop
            y, sm_lse, _ = fn(*inputs, **kwinputs)

            #if type(y) is tuple:
            #    y = y[0]            

        else:
            # Triton implementation does not support LSE backprop
            y = fn(*inputs, **kwinputs)

            if type(y) is tuple:
                y = y[0]

    qkv=inputs[0]
    if grad is None:
        if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:
            
            #q=qkv[:, 0], k=qkv[:, 1], v=qkv[:, 2]
            grad = (torch.randn_like(y),torch.randn(sm_lse.shape[0],sm_lse.shape[1],sm_lse.shape[2]))
        else:
            grad = torch.randn_like(y)
    else:
        if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:
            if grad[0].shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')
            if grad[1].shape != y.shape:
                raise RuntimeError('Grad shape does not match sm_lse shape')                
        else:        
            if grad.shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')

    if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:                
        for _ in range(repeats):  # warmup
            torch.autograd.grad((y.cuda(),sm_lse.cuda()), (inputs[0].cuda()), (grad[0].cuda(),grad[1].cuda()), allow_unused=True, retain_graph=True)
            #y.backward(grad[0], grad[1], retain_graph=True)
        t = benchmark.Timer(
                #stmt='y.backward(grad, retain_graph=True)',
                stmt='torch.autograd.grad((y.cuda(),sm_lse.cuda()), (inputs[0].cuda()), (grad[0].cuda(),grad[1].cuda()), allow_unused=True, retain_graph=True)',
                globals={'y': y, 'grad': grad, 'sm_lse': sm_lse, 'inputs': inputs},
                num_threads=torch.get_num_threads(),
            )
        m = t.timeit(repeats)
        if verbose:
            print(m)
        return t, m

    else:
        for _ in range(repeats):  # warmup
            #torch.autograd.grad((y.cuda(),sm_lse.cuda()), (inputs[0].cuda()), (grad[0].cuda(),grad[1].cuda()), allow_unused=True)
            y.backward(grad, retain_graph=True)
        t = benchmark.Timer(
                stmt='y.backward(grad, retain_graph=True)',
                #stmt='torch.autograd.grad((y.cuda(),sm_lse.cuda()), (inputs[0].cuda()), (grad[0].cuda(),grad[1].cuda()), allow_unused=True)',
                globals={'y': y, 'grad': grad},
                num_threads=torch.get_num_threads(),
            )
        m = t.timeit(repeats)
        if verbose:
            print(m)
        return t, m        


def benchmark_combined(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, amp=False,
                       amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """

    if verbose:
        print(desc, '- Forward + Backward pass')

    y=[]
    sm_lse=[]

    if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:
        y, sm_lse, _ = fn(*inputs, **kwinputs)
        #if type(y) is tuple:
        #    y = y[0]                
    else:            
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]

    def f(grad, *inputs, **kwinputs):
        y=[]
        sm_lse=[]

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:
                y, sm_lse, _ = fn(*inputs, **kwinputs)
                #if type(y) is tuple:
                #    y = y[0]                
            else:            
                y = fn(*inputs, **kwinputs)
                if type(y) is tuple:
                    y = y[0]

        if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:
            torch.autograd.grad((y.cuda(),sm_lse.cuda()), (inputs[0].cuda()), (grad[0].cuda(),grad[1].cuda()), allow_unused=True, retain_graph=True)
        else:
            y.backward(grad, retain_graph=True)

    qkv=inputs[0]
    if grad is None:
        if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:
            grad = (torch.randn_like(y),torch.randn(sm_lse.shape[0],sm_lse.shape[1],sm_lse.shape[2]))
        else:
            grad = torch.randn_like(y)
    else:
        if 'flash_attn_unpadded_qkvpacked_func' in fn.__name__:
            if grad[0].shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')
            if grad[1].shape != y.shape:
                raise RuntimeError('Grad shape does not match sm_lse shape')                
        else:        
            if grad.shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')

    for _ in range(repeats):  # warmup
        f(grad, *inputs, **kwinputs)
    t = benchmark.Timer(
            stmt='f(grad, *inputs, **kwinputs)',
            globals={'f': f, 'fn': fn, 'inputs': inputs, 'grad': grad, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_all(fn, *inputs, grad=None, repeats=10, desc='', verbose=True, amp=False,
                  amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    return (
        benchmark_forward(fn, *inputs, repeats=repeats, desc=desc, verbose=verbose,
                          amp=amp, amp_dtype=amp_dtype, **kwinputs),
        benchmark_backward(fn, *inputs, grad=grad, repeats=repeats, desc=desc, verbose=verbose,
                           amp=amp, amp_dtype=amp_dtype, **kwinputs),
        benchmark_combined(fn, *inputs, grad=grad, repeats=repeats, desc=desc, verbose=verbose,
                           amp=amp, amp_dtype=amp_dtype, **kwinputs),
    )


def pytorch_profiler(fn, *inputs, trace_filename=None, backward=False, amp=False,
                     amp_dtype=torch.float16, cpu=False, verbose=True, **kwinputs):
    """ Wrap benchmark functions in Pytorch profiler to see CUDA information. """
    if backward:
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            g = torch.randn_like(fn(*inputs, **kwinputs))
    for _ in range(30):   # Warm up
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            if backward:
                for x in inputs:
                    if isinstance(x, torch.Tensor):
                        x.grad = None
            # fn(*inputs, **kwinputs) if not backward else fn(*inputs, **kwinputs).backward(g)
            out = fn(*inputs, **kwinputs)
        # Backward should be done outside autocast
        if backward:
            out.backward(g)
    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            if backward:
                for x in inputs:
                    if isinstance(x, torch.Tensor):
                        x.grad = None
            out = fn(*inputs, **kwinputs)
        if backward: out.backward(g)
    if verbose:
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        print(prof.key_averages().table(row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)


def benchmark_memory(fn, *inputs, desc='', verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2 ** 20) * 1000)
    if verbose:
        print(f'{desc} max memory: {mem}GB')
    torch.cuda.empty_cache()
    return mem
