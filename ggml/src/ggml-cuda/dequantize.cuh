#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static const __device__ float TURBO3_CENTROIDS_D[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    const int j0 = iqs * 2;
    const int j1 = iqs * 2 + 1;

    uint8_t low2_0 = (x[ib].qs[j0 >> 2] >> ((j0 & 3) << 1)) & 0x3;
    uint8_t hi1_0  = (x[ib].signs[j0 >> 3] >> (j0 & 7)) & 0x1;
    uint8_t idx0   = low2_0 | (hi1_0 << 2);

    uint8_t low2_1 = (x[ib].qs[j1 >> 2] >> ((j1 & 3) << 1)) & 0x3;
    uint8_t hi1_1  = (x[ib].signs[j1 >> 3] >> (j1 & 7)) & 0x1;
    uint8_t idx1   = low2_1 | (hi1_1 << 2);

    v.x = TURBO3_CENTROIDS_D[idx0] * norm;
    v.y = TURBO3_CENTROIDS_D[idx1] * norm;
}

#define QR_TURBO3 2

static const __device__ float TURBO2_CENTROIDS_D[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static __device__ __forceinline__ void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    const int j0 = iqs * 2;
    const int j1 = iqs * 2 + 1;

    uint8_t idx0 = (x[ib].qs[j0 >> 2] >> ((j0 & 3) << 1)) & 0x3;
    uint8_t idx1 = (x[ib].qs[j1 >> 2] >> ((j1 & 3) << 1)) & 0x3;

    v.x = TURBO2_CENTROIDS_D[idx0] * norm;
    v.y = TURBO2_CENTROIDS_D[idx1] * norm;
}

#define QR_TURBO2 2

static __device__ __forceinline__ void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const float qjl_scale = 1.2533141f / 128.0f * rnorm;

    { const int j = iqs;
      int bo = j * 3, bi = bo / 8, bp = bo % 8;
      uint16_t raw = (uint16_t)x[ib].qs[bi];
      if (bi + 1 < 48) raw |= (uint16_t)x[ib].qs[bi + 1] << 8;
      uint8_t idx = (uint8_t)((raw >> bp) & 0x7);
      float s = (x[ib].signs[j / 8] & (1 << (j % 8))) ? 1.0f : -1.0f;
      v.x = (TURBO3_CENTROIDS_D[idx] + s * qjl_scale) * norm; }

    { const int j = iqs + 64;
      int bo = j * 3, bi = bo / 8, bp = bo % 8;
      uint16_t raw = (uint16_t)x[ib].qs[bi];
      if (bi + 1 < 48) raw |= (uint16_t)x[ib].qs[bi + 1] << 8;
      uint8_t idx = (uint8_t)((raw >> bp) & 0x7);
      float s = (x[ib].signs[j / 8] & (1 << (j % 8))) ? 1.0f : -1.0f;
      v.y = (TURBO3_CENTROIDS_D[idx] + s * qjl_scale) * norm; }
}

#define QR_TURBO4 2

