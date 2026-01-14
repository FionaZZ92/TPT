import ast
import argparse
import sys
import os
from typing import Dict, List
import triton
import triton.language as tl
from triton._C.libtriton import ir, passes

class PassOptIR:
    def __init__(self, source_file: str):
        self.source_file = source_file
    
    
    def generate_ttir_with_passes(self, output_file) -> None:
        if self.source_file == None:
            print("[ERROR] empty ttir code")
            return
        try:
            target = triton.runtime.driver.active.get_current_target()
            backend = triton.compiler.compiler.make_backend(target)
            codegen_fns = dict()
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
            passes.common.add_inliner(pm)
            passes.ttir.add_rewrite_tensor_pointer(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_combine(pm)
            passes.ttir.add_reorder_broadcast(pm)
            passes.common.add_cse(pm)
            passes.common.add_symbol_dce(pm)
            passes.ttir.add_loop_unroll(pm)
            print("===Executing Pass optimization...")
            pm.run(module)
            print("[DONE]Generate TTIR successfully.")
            with open(output_file, 'w') as f:
                f.write(str(module))
            print(f"[DONE] TTIR saved in: {output_file}")
        except Exception as e:
            print(f"[ERROR] Generate/pass TTIR failed: {e}")

    def generate_ttgir_with_passes(self, output_file) -> None:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,description="select test",)
    parser.add_argument("--source_file", "-s", type=str, default="")
    args = parser.parse_args()
    if not os.path.exists(args.source_file):
        print(f"[ERROR] {args.source_file} not exists.")
        sys.exit(1)
    obj = PassOptIR(args.source_file)
    obj.generate_ttir_with_passes("new.ttir")
