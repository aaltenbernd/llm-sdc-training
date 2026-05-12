#include <stdint.h>
#include <stdio.h>
#include <unordered_set>
#include <atomic>
#include <signal.h>
#include <random>
#include <chrono>
#include <unordered_map>
#include <algorithm>


#include "nvbit.h"

#define GET_VAR_INT_SILENT(var, env, def) { char* v = getenv(env); var = v ? atoi(v) : def; }

uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

int target_func;
int target_instr;

int count_func = 0;
int f_count = 0;

int target_smid;
int target_laneid;
int target_register;
uint32_t target_bitmask;
std::string target_op;
std::string target_func_contains;

std::unordered_set<CUfunction> already_instrumented;
static std::unordered_map<CUfunction, int> func_ids;

static std::atomic<bool> g_on{false};
static std::atomic<int> active_target_func{-1};

static void on_handler(int)  {
    g_on.store(true);
    GET_VAR_INT_SILENT(active_target_func, "TARGET_FUNC", -1);
}

static void off_handler(int) {
    g_on.store(false); 
}

void nvbit_at_init() {
    GET_VAR_INT(target_func, "TARGET_FUNC", -1, "Function ID to instrument (-1 = all)");
    GET_VAR_INT(target_instr, "TARGET_INSTR", -1, "Instruction index to instrument (-1 = all matching)");
    GET_VAR_INT(target_smid, "TARGET_SMID", -1, "SM ID for fault injection (-1 = all)");
    GET_VAR_INT(target_laneid, "TARGET_LANEID", -1, "Lane ID for fault injection (-1 = all)");
    GET_VAR_INT(target_register, "TARGET_REGISTER", 0, "Operand register index to corrupt");
    GET_VAR_INT(target_bitmask, "TARGET_BITMASK", 0, "Bitmask used to flip register bits");
    GET_VAR_STR(target_op, "TARGET_OP", "SASS opcode substring to match");
    GET_VAR_STR(target_func_contains, "TARGET_FUNC_CONTAINS", "Instrument functions containing this substring");

    GET_VAR_INT(instr_begin_interval, "INSTR_BEGIN", 0, "Inclusive lower bound for instruction index instrumentation");
    GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX, "Exclusive upper bound for instruction index instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbose NVBit logging");

    signal(SIGUSR1, on_handler);
    signal(SIGUSR2, off_handler);
}

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    related_functions.push_back(func);

    for (auto f : related_functions) {
        if (!already_instrumented.insert(f).second) { 
            continue; 
        }

        const char* fname = nvbit_get_func_name(ctx, f);

        if (!target_func_contains.empty()) {
            if (std::string(fname).find(target_func_contains) == std::string::npos) {
                continue;
            }
        }

        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
        if (verbose) {
            std::cout << "[NVBit Fault Injector] inspecting "
                << nvbit_get_func_name(ctx, f)
                << " - num instrs: " 
                << instrs.size()
                << " - count: "
                << count_func
                << std::endl
                << std::flush;
        }

        func_ids[f] = count_func;

        if (!(target_func < 0 || target_func == count_func)) {
            count_func++;
            continue;
        }

        std::vector<int> matching_idxs;

        for (auto instr : instrs) {
            uint32_t idx = instr->getIdx();

            if (idx < instr_begin_interval || idx >= instr_end_interval)
                continue;

            if (strstr(instr->getSass(), target_op.c_str())) {
                matching_idxs.push_back(idx);
            }
        }

        std::unordered_set<int> selected;

        if (target_instr >= 0) {
            for (int idx : matching_idxs) {
                if (idx == target_instr) {
                    selected.insert(idx);
                }
            }
        } else {
            for (int idx : matching_idxs) selected.insert(idx);
        }

        for (auto instr : instrs) {
            uint32_t idx = instr->getIdx();

            if (selected.find(idx) == selected.end())
                continue;

            if (target_register > 0) {
                nvbit_insert_call(instr, "insert_fault", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, instr->getOperand(target_register)->u.reg.num);
                nvbit_add_call_arg_const_val32(instr, target_smid);
                nvbit_add_call_arg_const_val32(instr, target_laneid);
                nvbit_add_call_arg_const_val32(instr, target_bitmask);
            }

            if (instr->getOperand(target_register)->u.reg.num != instr->getOperand(0)->u.reg.num) {
                nvbit_insert_call(instr, "insert_fault", IPOINT_AFTER);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, instr->getOperand(target_register)->u.reg.num);
                nvbit_add_call_arg_const_val32(instr, target_smid);
                nvbit_add_call_arg_const_val32(instr, target_laneid);
                nvbit_add_call_arg_const_val32(instr, target_bitmask);
            }

            if (verbose) { 
                std::cout << "[NVBit Fault Injector] " 
                    << instr->getSass() 
                    << "idx: " 
                    << instr->getIdx() 
                    << "; func: " 
                    << count_func 
                    << "\n" 
                    << std::endl 
                    << std::flush; 
            }
        }
        count_func++;
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            bool global_on = g_on.load(std::memory_order_relaxed);
            bool enable = false;
            if (global_on) {
                instrument_function_if_needed(ctx, func);
                auto it = func_ids.find(func);
                if (it != func_ids.end()) {
                    int fid = it->second;
                    enable = (fid == active_target_func.load());

                    if (enable && verbose) {
                        const char* fname = nvbit_get_func_name(ctx, func);
                        std::cout << "[NVBit] Enabling instrumentation for function ID "
                                << fid << ": " << fname << std::endl;
                    }
                }
            }

            nvbit_enable_instrumented(ctx, func, enable);
        }
    }
}