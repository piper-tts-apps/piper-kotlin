package dev.ihorshevchuk.piper.engine

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File
import kotlin.math.sqrt

/**
 * Baseline bench (Oct 2026, condition A of the Android streaming A/B).
 *
 * Measures the CURRENT main engine (piper1-gpl pin 404aefe, ORT graph
 * optimization BASIC, OnnxThreadPolicy threads) on a physical device,
 * before any pin bump or split/streaming change. Timing comes from the
 * timing-only PIPER_PERF lines in PiperEngine / SynthesisOrchestrator
 * (branch baseline/perf-timing-only) plus per-text lines printed here:
 * PIPER_BENCH result carries total samples, chunk count, time to first
 * chunk (synthesis-side, first onSamples callback), wall time, audio
 * length and RMS (non-silence correctness proxy; it cannot replace a
 * human listen).
 *
 * Models and espeak data are staged on the device beforehand:
 *   adb push <voices> /data/local/tmp/baseline-voices/
 *   adb push <staged>/espeak-ng-data /data/local/tmp/espeak-ng-data
 * Run only this class:
 *   ./gradlew :piper-engine:connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=dev.ihorshevchuk.piper.engine.BaselineBenchTest \
 *     -Pandroid.testInstrumentationRunnerArguments.espeakDataPath=/data/local/tmp/espeak-ng-data
 * (voices dir overridable via instrumentation arg baselineVoicesDir.)
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class BaselineBenchTest {

    companion object {
        // Fixed bench texts (baseline protocol, Oct 24 2026). B/C conditions
        // must reuse these verbatim.
        private const val TEXT_SHORT = "Home screen. Messages, double tap to open."
        private const val TEXT_MEDIUM =
            "The quick brown fox jumps over the lazy dog. Pack my box with " +
                "five dozen liquor jugs. How vexingly quick daft zebras jump!"
        private const val TEXT_LONG =
            "The history of the town began more than a hundred years ago, " +
                "when the first settlers arrived in the valley and built their " +
                "homes beside the river. Over the years the small settlement " +
                "grew into a busy community, with schools, shops, and a railway " +
                "station that connected it to the rest of the country. Today, " +
                "visitors come from far away to walk its quiet streets, visit " +
                "the old market square, and listen to stories passed down " +
                "through generations of families who still call this place " +
                "their home."

        private lateinit var espeakDataPath: String
        private lateinit var voicesDir: File

        @JvmStatic
        @BeforeClass
        fun setUpOnce() {
            val args = InstrumentationRegistry.getArguments()
            espeakDataPath = args.getString("espeakDataPath")
                ?: throw AssertionError("Missing instrumentation argument 'espeakDataPath'")
            require(File(espeakDataPath).isDirectory) {
                "espeak-ng data not found at $espeakDataPath"
            }
            voicesDir = File(args.getString("baselineVoicesDir") ?: "/data/local/tmp/baseline-voices")
            require(voicesDir.isDirectory) { "baseline voices not found at $voicesDir" }
        }
    }

    @Test(timeout = 900_000)
    fun bench1AmyFp32() {
        runBench("fp32", "en_US-amy-medium-fp32.onnx")
    }

    @Test(timeout = 900_000)
    fun bench2AmyFp16() {
        runBench("fp16", "en_US-amy-medium-fp16.onnx")
    }

    private fun runBench(precision: String, modelName: String) {
        val model = File(voicesDir, modelName)
        val config = File(voicesDir, "$modelName.json")
        require(model.isFile && config.isFile) { "missing $model or $config" }
        println(
            "PIPER_BENCH begin precision=$precision model=$modelName bytes=${model.length()}"
        )
        val engine = newEngine(model, config)
        // Warm-up synthesis (discarded); it triggers engine creation, so the
        // PIPER_PERF create line printed during it is this engine's create.
        synthOnce(engine, "Warm up.", "warmup", precision, 0)
        for (run in 1..2) {
            synthOnce(engine, TEXT_SHORT, "short", precision, run)
            synthOnce(engine, TEXT_MEDIUM, "medium", precision, run)
            synthOnce(engine, TEXT_LONG, "long", precision, run)
        }
        engine.close()
        // Second create in the same process (native libraries already
        // loaded): a warm re-create sample next to the first create above.
        println("PIPER_BENCH phase=recreate precision=$precision")
        val engine2 = newEngine(model, config)
        synthOnce(engine2, TEXT_SHORT, "short-recreate", precision, 0)
        engine2.close()
        println("PIPER_BENCH end precision=$precision")
    }

    private fun newEngine(model: File, config: File): PiperEngine =
        PiperEngine(
            PiperCreateOptions(
                modelPath = model.absolutePath,
                configPath = config.absolutePath,
                espeakDataPath = espeakDataPath
            )
        )

    private fun synthOnce(
        engine: PiperEngine,
        text: String,
        label: String,
        precision: String,
        run: Int
    ) {
        var firstChunkNs = 0L
        var chunks = 0
        var totalSamples = 0L
        var sumSquares = 0.0
        val startNs = System.nanoTime()
        engine.synthesize(
            text,
            onSamples = { samples ->
                if (firstChunkNs == 0L) firstChunkNs = System.nanoTime()
                chunks++
                totalSamples += samples.size
                for (v in samples) sumSquares += (v * v).toDouble()
            }
        )
        val wallMs = (System.nanoTime() - startNs) / 1_000_000
        val ttfaMs = if (firstChunkNs > 0L) (firstChunkNs - startNs) / 1_000_000 else -1L
        val sampleRate = engine.currentSampleRate.get()
        val audioMs = if (sampleRate > 0) totalSamples * 1000L / sampleRate else 0L
        val rms = if (totalSamples > 0L) sqrt(sumSquares / totalSamples) else 0.0
        println(
            "PIPER_BENCH result precision=$precision run=$run text=$label " +
                "samples=$totalSamples sampleRate=$sampleRate chunks=$chunks " +
                "ttfaMs=$ttfaMs wallMs=$wallMs audioMs=$audioMs " +
                "rms=${"%.5f".format(rms)}"
        )
        assertTrue("no audio produced for $label", totalSamples > 1000L)
        assertTrue("audio looks silent for $label (rms=$rms)", rms > 0.001)
    }
}
