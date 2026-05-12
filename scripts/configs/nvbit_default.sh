#!/bin/bash

export LD_PRELOAD=./nvbit/1.7.4_nvbit_release/tools/fault_injection/fault_injection.so

export TOOL_VERBOSE=0

export TARGET_REGISTER=1
export TARGET_OP=HMMA

export TARGET_SMID=0
export TARGET_LANEID=0

export TARGET_BITMASK=4096

export TARGET_FUNC=-1
export TARGET_INSTR=-1

export TARGET_FUNC_CONTAINS=gemm