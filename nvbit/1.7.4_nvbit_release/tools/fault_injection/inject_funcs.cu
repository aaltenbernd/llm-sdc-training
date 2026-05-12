#include "nvbit_reg_rw.h"

extern "C" __device__ __noinline__ void insert_fault(
                                        int pred,
                                        int dst_reg,
                                        int target_smid,
                                        int target_laneid,
                                        int target_bitmask) {

    if (!pred) return;

    if (target_smid >= 0) {

        uint32_t smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));

        if (smid != (uint32_t)target_smid)
            return;
    }

    if (target_laneid >= 0) {

        uint32_t laneid;
        asm("mov.u32 %0, %laneid;" : "=r"(laneid));

        if (laneid != (uint32_t)target_laneid)
            return;
    }

    uint32_t out = nvbit_read_reg(dst_reg) ^ target_bitmask;
    nvbit_write_reg(dst_reg, out);
}