package dev.ihorshevchuk.piper.engine

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.sqrt

/**
 * Split-only bench (Oct 2026, app-integration phase of the Android
 * streaming work).
 *
 * Question this answers before the app adopts splits-only storage:
 * does engine creation succeed when the model file itself is ABSENT
 * and only the config plus the encoder/decoder pair sit next to the
 * model path? The native layer's contract says the pair replaces the
 * model file wholesale; this test proves it on-device, with the
 * PiperEngine file guard relaxed to accept a complete pair.
 *
 * Voices dir layout (staged beforehand, NO .onnx model file):
 *   en_US-amy-medium-fp16.onnx.json
 *   en_US-amy-medium-fp16.encoder.onnx
 *   en_US-amy-medium-fp16.decoder.onnx
 * Proof of streaming is the chunk count: monolithic synthesis of these
 * texts produced chunk counts equal to clause counts (short 2,
 * medium 3, long 6); split synthesis must produce strictly more.
 */
@RunWith(AndroidJUnit4::class)
class SplitOnlyBenchTest {

    companion object {
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
            voicesDir = File(
                args.getString("splitOnlyVoicesDir") ?: "/data/local/tmp/streaming-only-voices"
            )
            require(voicesDir.isDirectory) { "split-only voices not found at $voicesDir" }
        }
    }

    @Test(timeout = 900_000)
    fun benchAmyFp16SplitOnly() {
        val stem = "en_US-amy-medium-fp16"
        val model = File(voicesDir, "$stem.onnx")
        val config = File(voicesDir, "$stem.onnx.json")
        assertFalse("model file must be absent for this test", model.isFile)
        require(config.isFile) { "missing $config" }
        require(File(voicesDir, "$stem.encoder.onnx").isFile) { "missing encoder" }
        require(File(voicesDir, "$stem.decoder.onnx").isFile) { "missing decoder" }
        println("PIPER_BENCH begin precision=fp16-split-only model=${model.name} (absent)")

        val engine = PiperEngine(
            PiperCreateOptions(
                modelPath = model.absolutePath,
                configPath = config.absolutePath,
                espeakDataPath = espeakDataPath
            )
        )
        try {
            synthAndAssert(engine, "Warm up.", "warmup", 0)
            val short = synthAndAssert(engine, TEXT_SHORT, "short", 1)
            val medium = synthAndAssert(engine, TEXT_MEDIUM, "medium", 2)
            val long = synthAndAssert(engine, TEXT_LONG, "long", 3)
            assertTrue("short chunks $short must exceed monolithic 2", short > 2)
            assertTrue("medium chunks $medium must exceed monolithic 3", medium > 3)
            assertTrue("long chunks $long must exceed monolithic 6", long > 6)
        } finally {
            engine.close()
        }
        println("PIPER_BENCH end precision=fp16-split-only")
    }

    private fun synthAndAssert(engine: PiperEngine, text: String, label: String, run: Int): Int {
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
            "PIPER_BENCH result precision=fp16-split-only run=$run text=$label " +
                "samples=$totalSamples sampleRate=$sampleRate chunks=$chunks " +
                "ttfaMs=$ttfaMs wallMs=$wallMs audioMs=$audioMs " +
                "rms=${"%.5f".format(rms)}"
        )
        assertTrue("no audio produced for $label", totalSamples > 1000L)
        assertTrue("audio looks silent for $label (rms=$rms)", rms > 0.001)
        return chunks
    }
}
