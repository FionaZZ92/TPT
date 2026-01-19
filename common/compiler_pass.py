import ast
import argparse
import sys
import os
from typing import Dict, List
import triton
import triton.language as tl
from triton._C.libtriton import ir, passes

class PassOptIR:
    def __init__(self, args):
        self.source_file = args.source_file
        self.matrix_instr_nonkdim = args.matrix_instr_nonkdim
        self.kpack = args.kpack
        self.num_warps = args.num_warps
        self.num_stages = args.num_stages
    
    
    def generate_ttir_with_passes(self, output_file) -> None:
        if self.source_file == None:
            print("[ERROR] empty ttir code")
            return
        try:
            target = triton.runtime.driver.active.get_current_target()
            backend = triton.compiler.compiler.make_backend(target)
            #codegen_fns = dict()
            module_map = backend.get_module_map()
            context = ir.context()
            ir.load_dialects(context)
            backend.load_dialects(context)
            module = ir.parse_mlir_module(self.source_file, context)
            #module = self.src.make_ir(target, options, codegen_fns, module_map, context)
            #module = ir.module("", context) #triton not provide py interface for module

            # create pass manager
            pm = ir.pass_manager(context)
            pm.enable_debug()
            
            print("===Applying Pass...")
            #print(f"passes.common: {dir(passes.common)}")
            #print(f"passes.ttir: {dir(passes.ttir)}")
            passes.common.add_inliner(pm)
            #重写张量指针（tensor pointer）操作‌，以优化内存访问模式并提高性能
            passes.ttir.add_rewrite_tensor_pointer(pm)
            #将涉及张量描述符（tensor descriptor）的操作重写为使用张量指针（tensor pointer）的特定模式‌
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
            #标准化和规范化，如x*1->x
            passes.common.add_canonicalizer(pm)
            #tt.dot 操作和 arith.addf 操作融合
            passes.ttir.add_combine(pm)
            #broadcast和splat操作移动到elementwise操作的后面
            passes.ttir.add_reorder_broadcast(pm)
            #公共子表达式消除
            passes.common.add_cse(pm)
            #静态单赋值形式下的公共子表达式消除
            passes.common.add_sccp(pm)
            #循环感知的公共子表达式消除
            passes.ttir.add_loop_aware_cse(pm)
            #循环不变量外提
            passes.ttir.add_triton_licm(pm)
            #死代码消除‌
            passes.common.add_symbol_dce(pm)
            #循环展开
            passes.ttir.add_loop_unroll(pm)
            print("===Executing Pass optimization...")
            pm.run(module)
            print("[DONE]Generate TTIR successfully.")
            with open(output_file, 'w') as f:
                f.write(str(module))
            print(f"[DONE] TTIR saved in: {output_file}")

        except Exception as e:
            print(f"[ERROR] Generate/pass TTIR failed: {e}")



    def generate_ttgir_with_passes(self, output_file, config) -> None:
        if self.source_file == None:
            print("[ERROR] empty ttir code")
            return
        try:
            target = triton.runtime.driver.active.get_current_target()
            backend = triton.compiler.compiler.make_backend(target)
            module_map = backend.get_module_map()
            context = ir.context()
            ir.load_dialects(context)
            backend.load_dialects(context)
            module = ir.parse_mlir_module(self.source_file, context)

            # create pass manager
            pm = ir.pass_manager(context)
            pm.enable_debug()
            
            passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{target.arch}", self.num_warps, target.warp_size, 1) #1 CTA for kernel execution 
            pm.run(module, 'make_ttgir_stage1')

            pm = ir.pass_manager(context)
            pm.enable_debug()

            print("===Applying Pass...")

            passes.ttgpuir.add_coalesce(pm)
            passes.ttgpuir.add_f32_dot_tc(pm, False)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_optimize_thread_locality(pm)
            #For GEMM kernels on an AMD MI300X accelerator, mfma_16x16 typically outperforms mfma_32x32, even for large tile/GEMM sizes.
            amd.passes.ttgpuir.add_accelerate_matmul(pm, target.arch, self.matrix_instr_nonkdim, self.kpack)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            amd.passes.ttgpuir.add_optimize_epilogue(pm)
            amd.passes.ttgpuir.add_optimize_dot_operands(pm, target.arch)
            amd.passes.ttgpuir.add_hoist_layout_conversions(pm)
            amd.passes.ttgpuir.add_sink_layout_conversions(pm)

            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)

            amd.passes.ttgpuir.add_schedule_loops(pm, 2) #num_stages is 2, 3 for Nv
            passes.common.add_canonicalizer(pm)
            #try scheduler hint later
            # for hint in str(""attention,memory-bound-attention").split(","):
            #    amd.passes.ttgpuir.insert_instruction_sched_hints(pm, hint)
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.ttgpuir.add_reduce_data_duplication(pm)

            if target.arch == "gfx942":
                amd.passes.ttgpuir.add_pipeline(pm, False, True) 
                #特定的张量转置操作（transpose）优化为线程级（in-thread）操作
                amd.passes.ttgpuir.add_in_thread_transpose(pm)
                #移除不必要的layout转换操作
                passes.ttgpuir.add_remove_layout_conversions(pm)
            elif target.arch == "gfx950":
                amd.passes.ttgpuir.add_pipeline(pm, True, True) #use_async_copy & use_block_pingpong
                amd.passes.ttgpuir.add_coalesce_async_copy(pm, target.arch)
            else:
                amd.passes.ttgpuir.add_pipeline(pm, False, False) 
            amd.passes.ttgpuir.add_reorder_instructions(pm)
            amd.passes.ttgpuir.add_block_pingpong(pm, self.num_stages)

            amd.passes.ttgpuir.add_fold_true_cmpi(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            passes.common.add_symbol_dce(pm)
            print("===Executing Pass optimization...")
            pm.run(module)
            print("[DONE]Generate TTGIR successfully.")
            with open(output_file, 'w') as f:
                f.write(str(module))
            print(f"[DONE] TTGIR saved in: {output_file}")

        except Exception as e:
            print(f"[ERROR] Generate/pass TTGIR failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,description="select test",)
    parser.add_argument("--source_file", "-s", type=str, default="")
    parser.add_argument("--ttgir", "-ttg", action="store_true")
    parser.add_argument("--num_warps", "-nw", type=int, default=4)
    parser.add_argument("--num_stages", "-ns", type=int, default=2)
    parser.add_argument("--matrix_instr_nonkdim", "-min", type=int, default=16)
    parser.add_argument("--kpack", type=int, default=1)
    args = parser.parse_args()
    if not os.path.exists(args.source_file):
        print(f"[ERROR] {args.source_file} not exists.")
        sys.exit(1)
    obj = PassOptIR(args)
    if args.ttgir:
        obj.generate_ttgir_with_passes("new.ttgir")
    else:
        obj.generate_ttir_with_passes("new.ttir")
