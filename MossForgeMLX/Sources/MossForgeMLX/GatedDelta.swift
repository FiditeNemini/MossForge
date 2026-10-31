import Foundation
import MLX

// The Metal kernel source below is adapted from mlx-lm `mlx_lm/models/gated_delta.py`
// (`_make_gated_delta_kernel`, scalar gate, no mask) — Copyright © 2025 Apple Inc., MIT licence.
// See NOTES.md.

/// The gated delta rule, run as a recurrence over time (the prefill form the encoder needs):
///
/// ```
/// S_t = g_t · S_{t−1}                      (per value head, S ∈ ℝ^{Dv×Dk})
/// S_t += k_t ⊗ β_t (v_t − S_t k_t)
/// y_t = S_t q_t
/// ```
///
/// One thread group per (batch, value head, Dv row-block); each SIMD lane holds Dk/32 state
/// columns, so state never leaves registers. GQA: value head `h` reads key head `h / (Hv/Hk)`
/// (= torch `repeat_interleave`). Callers l2-normalise q/k and fold `Dk^-½` into q.
public enum GatedDelta {
    private static let kernel = MLXFast.metalKernel(
        name: "moss_gated_delta_step",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
                auto n = thread_position_in_grid.z;
                auto b_idx = n / Hv;
                auto hv_idx = n % Hv;
                auto hk_idx = hv_idx / (Hv / Hk);
                constexpr int n_per_t = Dk / 32;

                // q, k: [B, T, Hk, Dk]
                auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
                auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

                // v, y: [B, T, Hv, Dv]
                auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
                y += b_idx * T * Hv * Dv + hv_idx * Dv;

                auto dk_idx = thread_position_in_threadgroup.x;
                auto dv_idx = thread_position_in_grid.y;

                // state_in, state_out: [B, Hv, Dv, Dk]
                auto i_state = state_in + (n * Dv + dv_idx) * Dk;
                auto o_state = state_out + (n * Dv + dv_idx) * Dk;

                float state[n_per_t];
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = static_cast<float>(i_state[s_idx]);
                }

                // g, beta: [B, T, Hv]
                auto g_ = g + b_idx * T * Hv;
                auto beta_ = beta + b_idx * T * Hv;

                for (int t = 0; t < T; ++t) {
                  float kv_mem = 0.0f;
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * g_[hv_idx];
                    kv_mem += state[i] * k_[s_idx];
                  }
                  kv_mem = simd_sum(kv_mem);

                  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                  float out = 0.0f;
                  for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] + k_[s_idx] * delta;
                    out += state[i] * q_[s_idx];
                  }
                  out = simd_sum(out);
                  if (thread_index_in_simdgroup == 0) {
                    y[dv_idx] = static_cast<InT>(out);
                  }
                  q_ += Hk * Dk;
                  k_ += Hk * Dk;
                  v_ += Hv * Dv;
                  y += Hv * Dv;
                  g_ += Hv;
                  beta_ += Hv;
                }
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  o_state[s_idx] = static_cast<StT>(state[i]);
                }
            """)

    /// - Parameters: q, k `[B,T,Hk,Dk]`; v `[B,T,Hv,Dv]`; g (decay multiplier, not its log) and
    ///   beta `[B,T,Hv]`. Dk must be a multiple of 32.
    /// - Returns: y `[B,T,Hv,Dv]` (q's dtype) and the final F32 state `[B,Hv,Dv,Dk]`.
    public static func run(q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray? = nil)
        -> (y: MLXArray, state: MLXArray)
    {
        let (b, t, hk, dk) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3))
        let (hv, dv) = (v.dim(2), v.dim(3))
        precondition(dk % 32 == 0 && hv % hk == 0, "gated delta: Dk % 32 and Hv % Hk must be 0")
        let s0 = state ?? MLXArray.zeros([b, hv, dv, dk], type: Float.self)
        let out = kernel(
            [q, k, v.asType(q.dtype), g.asType(q.dtype), beta.asType(q.dtype), s0, t],
            template: [("InT", q.dtype), ("StT", DType.float32), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)],
            grid: (32, dv, b * hv), threadGroup: (32, 4, 1),
            outputShapes: [[b, t, hv, dv], [b, hv, dv, dk]], outputDTypes: [q.dtype, .float32])
        return (out[0], out[1])
    }

    /// Ops-only reference recurrence (slow; one graph node per step) for checking the kernel.
    public static func reference(q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray) -> MLXArray {
        let rep = v.dim(2) / k.dim(2)
        let (qq, kk) = rep > 1 ? (repeated(q, count: rep, axis: 2), repeated(k, count: rep, axis: 2)) : (q, k)
        var state = MLXArray.zeros([v.dim(0), v.dim(2), v.dim(3), k.dim(3)], type: Float.self)
        var ys: [MLXArray] = []
        for t in 0..<v.dim(1) {
            let (qt, kt, vt) = (qq[0..., t].asType(.float32), kk[0..., t].asType(.float32), v[0..., t].asType(.float32))
            state = state * g[0..., t].asType(.float32)[0..., 0..., .newAxis, .newAxis]
            let kvMem = (state * kt[0..., 0..., .newAxis, 0...]).sum(axis: -1)
            let delta = (vt - kvMem) * beta[0..., t].asType(.float32)[0..., 0..., .newAxis]
            state = state + kt[0..., 0..., .newAxis, 0...] * delta[0..., 0..., 0..., .newAxis]
            ys.append((state * qt[0..., 0..., .newAxis, 0...]).sum(axis: -1))
        }
        return stacked(ys, axis: 1)
    }
}
