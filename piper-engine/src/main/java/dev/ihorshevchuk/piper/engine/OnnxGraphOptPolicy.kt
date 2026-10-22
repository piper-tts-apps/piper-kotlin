package dev.ihorshevchuk.piper.engine

/**
 * Selects the ONNX Runtime graph optimization level the native engine
 * should request (performance spike, Oct 2026).
 *
 * Pinned piper1-gpl hard-codes `ORT_ENABLE_BASIC`. The native patch
 * (0003) lets the caller ask for `ORT_ENABLE_ALL` instead - full constant
 * folding and node fusions - via `piper_create_options.ort_graph_opt_all`,
 * filled in by the JNI layer from this policy, mirroring how
 * [OnnxThreadPolicy] feeds `ort_intra_op_num_threads`.
 *
 * Spike setting: ALL. The baseline build (main) requests BASIC, so an
 * A/B device run only needs the two builds installed in turn. Watch the
 * INT8 (Compact) voices in particular: under older ONNX Runtime versions
 * some graph optimizations had to be disabled for INT8 models (the
 * superseded PR #4 workaround); ORT 1.31 is expected to be past that,
 * but the spike must prove it per precision, not assume it.
 */
object OnnxGraphOptPolicy {
    fun useAllOptimizations(): Boolean = true
}
