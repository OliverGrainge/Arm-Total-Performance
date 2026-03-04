# Tutorial 3: Using ATP Instruction Mix to Optimise a C++ Workload

This tutorial shows you how to use **Arm Total Performance (ATP)** to optimise a real C++ LLM chatbot program. You will apply two ATP recipes - **CPU Cycle Hotspots** and **Instruction Mix** - in sequence. CPU Cycle Hotspots tells you which function is worth investigating. Instruction Mix tells you how that function is spending its time at the instruction level. Together they give you a concrete, evidence-based optimisation direction rather than a guess.

The workload is `gpt2`, a text-generation program that runs a medium-sized language model on the CPU. It is a realistic inference workload: compute-intensive, numerically dominated, and representative of the kind of code that benefits most from Arm's vector extensions. The tutorial follows the full loop a developer would use in practice: profile the workload, diagnose the bottleneck, apply a targeted fix, and re-profile to confirm the change worked. By the end of this tutorial, you will know how to:

1. Use CPU Cycle Hotspots to determine which function is worth investigating.
2. Use the Instruction Mix recipe to measure how instructions are distributed across scalar and vector units.
3. Use instruction-level evidence to decide whether a library-based implementation is likely to help.
4. Verify an optimisation by comparing instruction mix and throughput before and after.

## Before you begin

- An AWS Graviton3 instance (SVE is required for the optimised binary)
- GCC 11+ or Clang 14+
- CMake 3.16+
- Python 3.8+ with `pip`
- ATP installed and configured

## Background: Scalar versus Vector Arithmetic on Graviton3

Instruction mix matters because performance depends not just on how many instructions a program executes, but on what kind of instructions they are. On a floating-point workload, a loop that mainly uses scalar instructions will do less work per instruction than one using the CPU's vector units. That makes instruction mix a useful signal to investigate: it shows whether the hot code is actually using the hardware efficiently, or leaving throughput on the table.

ATP's **Instruction Mix** recipe counts every retired instruction and groups it by type: scalar integer, scalar FP, NEON, SVE, load/store, branch, and so on. For this tutorial, the most important categories are:

- **Scalar FP**: floating-point instructions operating on one value at a time. A simple non-vectorised C loop will often compile to this.
- **NEON**: Arm's 128-bit SIMD extension. For single-precision data, one instruction can operate on 4 floats at once.
- **SVE**: Arm's Scalable Vector Extension. On Graviton3, SVE is 256 bits wide, so one instruction can operate on 8 single-precision floats at once.

For a compute-heavy kernel, the balance between Scalar FP and SVE is a strong diagnostic signal. If the hot numerical loop is mostly scalar on a Graviton3 system, the processor's wider vector hardware is not being used well.


> **Note on measurement bias:** Because `matmul` accounts for ~85% of cycles in this workload, its instruction behaviour dominates the whole-program Instruction Mix ATP reports. The whole-program view is, in practice, a close proxy for what `matmul` is doing.

---

## The Workload

The program, `gpt2`, is a text generation engine. Given a short prompt, it generates new text one token at a time:

```text
$ ./gpt2 --model gpt2-medium "Once upon a time"
Once upon a time there was a man who had a great deal of money...
[200 tokens, 3.2 tok/s]
```

> **About GPT-2:** GPT-2 ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) was one of the most influential early language models, demonstrating that large-scale unsupervised training could produce surprisingly coherent text. By modern standards it is small, so do not expect ChatGPT-quality output.

A **token** is a chunk of text processed by the model, often a whole word, part of a word, or punctuation. `tok/s` means **tokens per second**, which is the standard throughput metric for text generation: higher `tok/s` means the model is generating text faster.

Internally, GPT-2 Medium is a large numerical model with 345 million stored floating-point weights. These weights are the learned numerical parameters that determine how the model transforms one sequence of tokens into the next. Generating each new token requires running those weights through 24 layers of repeated arithmetic. Do not worry if you do not understand LLM internals; for this tutorial, the important point is that this is a compute-heavy C++ workload with several functions involved in the generation path. The next step is to download the model data and convert it into the binary format expected by the C++ program, then build and run the baseline workload.

### Build and Run

Export the model weights (requires an internet connection on the first run):

```bash
pip3 install torch transformers
python3 src/export_gpt2.py --model gpt2-medium
```

This downloads the GPT-2 Medium parameters and writes them into `models/gpt2-medium/weights.bin` and `models/gpt2-medium/vocab.bin`, the binary files used by the C++ program to run the text generation.

Next, build the C++ program:

```bash
cmake -S . -B build
cmake --build build --parallel
```

Run the baseline and record the throughput:

You can change the prompt `"Once upon a time"` if you want. The `-n` parameter sets the maximum number of tokens to generate.

```bash
cd build
./gpt2 --model gpt2-medium "Once upon a time" -n 50
```

When the program finishes generating, it prints a final line like this showing the generation throughput in tokens per second. Write it down; this is your baseline measurement. Next, you will use ATP to identify where the program spends its time so you can start improving its performance.

```text
[50 tokens, 17.2837 tok/s]
```

<img src="assets/gpt_textgen.gif" width="850" alt="Terminal recording of gpt2 generating text and printing its throughput in tokens per second"/>

---

## Profile the Baseline: CPU Cycle Hotspots

The first question to ask about any workload is: where does the program actually spend its time? Without this, any optimisation attempt is just a guess. CPU Cycle Hotspots uses hardware performance counters to attribute cycles to individual functions as the program runs.

### Step 1: Run the recipe

Open ATP and select **Recipes -> CPU Cycle Hotspots**. Set the workload to launch `gpt2` with arguments `--model gpt2-medium "Once upon a time" -n 100`, then click **Run Recipe**.

### Step 2: Read the flame graph

When the run completes, select the **CPU Hotspots** result to view the **Flame Graph**. Each horizontal bar represents a function. Bars stacked on top of a function are routines it calls, so the graph shows the call stack from caller to callee. Width is proportional to the CPU time spent in that function and everything it calls. Wider bars near the bottom of the graph are the places the CPU is actually executing, so they are the candidates worth investigating.

<img src="assets/gpt2_flamegraph.png" width="850" alt="CPU Cycle Hotspots flame graph for gpt2: a single bar dominates the execution time"/>

The flame graph shows that most of the program's time is spent in `forward`, and most of that time is inside the `matmul` calls made by `forward`. Now, in the next step, let's quantify exactly how much time is spent where. 

### Step 3: Read the Functions table

In the CPU Hotspots run, switch to the **Functions** tab. Here, ATP lists every sampled function alongside its percentage of total cycles that it occupies:

<img src="assets/gpt2_functions_table.png" width="850" alt="Functions table for gpt2 showing matmul at roughly 83% of cycles with all other functions combined as a small fraction"/>

The profile is clear: `matmul` accounts for about 84% of runtime. That means improving `matmul` should translate directly into higher throughput, but only up to a point: the other ~16% of the program still remains. Even if `matmul` were made infinitely fast, the total speedup would still be capped at about 6.25x.

**Your diagnosis so far:** `matmul` accounts for ~85% of cycles - the clear target. Now the question changes. We know *where* the time goes. The next question is: *how* is `matmul` spending it? Is the hardware being used effectively, or is there capacity being wasted?

---

## Profile the Baseline: Instruction Mix

ATP's Instruction Mix recipe answers this question directly. It counts every instruction that retires and classifies it by type. For this workload, the most important categories are Scalar FP and SVE. If the hot `matmul` loop emits scalar instructions, the CPU processes one float per instruction. If it emits SVE instructions, it can process 8 floats per instruction. The ratio between these categories is therefore a useful signal for whether the implementation is making good use of the processor's arithmetic hardware.

### Step 1: Run the recipe

In ATP, select **Recipes -> Instruction Mix**. Use the same workload and arguments - `gpt2 --model gpt2-medium "Once upon a time" -n 100` - then click **Run Recipe**.

### Step 2: Read the breakdown

ATP presents a workload-wide breakdown of retired instruction types. Because `matmul` accounts for about 85% of cycles, this chart is effectively showing you how `matmul` executes:

<img src="assets/gpt2_instruction_mix.png" width="850" alt="Instruction Mix for gpt2: Scalar FP accounts for the largest share of retired instructions and the SVE row reads 0%"/>

Two things stand out in the chart. First, the largest categories are **Integer** and **Load**. That is what you would expect from a scalar matrix multiply: every iteration must compute addresses and indices, then load `row[j]` and `x[j]` before doing the arithmetic. Second, **Advanced SIMD** and **SVE** are at 0%, so the compiler has not vectorised the dominant loop.

The other visible bars match the structure of a simple scalar matrix multiply:

- **Integer**: loop counters and address calculations (`i`, `j`, and `row`) generate a large number of integer instructions.
- **Loads**: each inner-loop iteration reads `row[j]` and `x[j]`, so scalar loads are also a large share of the mix.
- **Scalar FP**: each `acc += row[j] * x[j]` is a scalar floating-point multiply-accumulate, but it is only one part of the loop body.
- **Branch**: the `for` loop conditions (`j < n_in`, `i < n_out`) and the `b ?` ternary each produce branch instructions.
- **SVE / Advanced SIMD**: **0%**. No NEON or SVE instructions appear in the hot path.

This matches the scalar `matmul` implementation in `src/gpt2.cpp`:

```cpp
static void matmul(float *out, const float *x, const float *W, const float *b,
                   int n_in, int n_out) {
    for (int i = 0; i < n_out; i++) {
        float acc = b ? b[i] : 0.f;
        const float *row = W + (size_t)i * n_in;
        for (int j = 0; j < n_in; j++) acc += row[j] * x[j];
        out[i] = acc;
    }
}
```

The inner loop performs one scalar multiply-accumulate per iteration, surrounded by scalar loads and index arithmetic, which is exactly the execution pattern the Instruction Mix chart is showing.


**Complete diagnosis:** the hot `matmul` function is executing as a scalar loop with scalar loads, index logic, and scalar floating-point operations. `SVE = 0%`, so Graviton's vector units are idle for the dominant operation.

---

## Fix and Re-Profile: KleidiAI SVE Microkernel

ATP has identified the problem: the dominant function is a scalar loop running on a processor with idle SVE vector units. The fix is to replace it with a vectorised implementation from [KleidiAI](https://github.com/ARM-software/kleidiai), Arm's open-source library of hand-tuned AI microkernels.

The change has two parts. First, weight matrices are repacked once at startup into a tiled memory layout the microkernel can load efficiently - this cost is not included in the reported tok/s. Second, the scalar loop body is replaced by calls to `ukernel.run_matmul`:

```cpp
static void matmul(float* out, const float* x, const uint8_t* rhs_packed,
                   int n_in, int n_out)
{
    const size_t m = 1, k = (size_t)n_in;
    const size_t lhs_stride = k * sizeof(float);
    const size_t dst_stride_row = (size_t)n_out * sizeof(float);
    const size_t dst_stride_col = sizeof(float);
    const size_t n_step = ukernel.get_n_step();

    const size_t n_blocks = ((size_t)n_out + n_step - 1) / n_step;

    for (size_t b = 0; b < n_blocks; b++) {
        const size_t n_start = b * n_step;
        const size_t rhs_offset = ukernel.get_rhs_packed_offset(n_start, k);

        ukernel.run_matmul(
            m, n_step, k,
            x, lhs_stride,
            rhs_packed + rhs_offset,
            out + n_start, dst_stride_row, dst_stride_col,
            -FLT_MAX, FLT_MAX
        );
    }
}
```

The algorithm, model weights, and generated text are all unchanged. The only difference is what instructions the CPU executes. See the [KleidiAI repository](https://github.com/ARM-software/kleidiai) for details on the available kernels and their target microarchitectures.

### Step 1: Re-profile with Instruction Mix

Run the Instruction Mix recipe again, this time with `gpt2_kai_sve` as the workload. Always re-profile after a change and never assume your optimisation had the intended effect.

In ATP, select **Recipes -> Instruction Mix**, set the workload to `gpt2_kai_sve --model gpt2-medium "Once upon a time" -n 100`, and click **Run Recipe**.

### Step 2: Read the new Instruction Mix breakdown

<p align="center">
<img src="assets/gpt2_kai_sve_instruction_mix.png" width="500" alt="Instruction Mix for gpt2_kai_sve: SVE instructions now account for the largest share of retired instructions at roughly 53%, with Floating Point Operations near zero"/>
</p>

The chart has changed dramatically. **SVE Operations** are now the single largest category at roughly 53% of all retired instructions - up from 0% in the baseline. **Floating Point Operations** have collapsed to near zero. This is the inversion the diagnosis predicted: the processor is now spending the majority of its arithmetic work in 256-bit vector instructions rather than scalar ones.

The **Load Operations** bar (~34%) is higher than in the baseline. This is a direct side-effect of the packed weight layout: the microkernel loads a wide tile of weight data per inner-loop iteration, so loads are a larger fraction of the total instruction stream even though the absolute number of loads has gone down.

**Integer Operations** (~18%) and **Branch** (~7%) have both decreased relative to the baseline. The scalar loop generated a large number of index and counter instructions for every element; the tiled microkernel amortises that overhead across a whole tile of outputs, so far fewer integer and branch instructions are needed per unit of useful work.

The gap that the baseline profile revealed - SVE at zero on a compute-heavy workload - has been closed.

---

## Verify with Throughput

Compare the tok/s figures from both runs:

| Binary | Throughput | SVE % | Scalar FP % |
|---|---:|---:|---:|
| `gpt2` | 3.4 tok/s | 0% | ~0% |
| `gpt2_kai_sve` | 18.0 tok/s | ~54% | ~0% |
| Improvement | **5.3×** | - | - |

The throughput increase is the real-world consequence of the instruction-level change ATP measured. No algorithm changed, no data was restructured, no compiler flags were added. The improvement comes entirely from replacing scalar FP instructions with SVE FP instructions in the one function that dominates the runtime.

<p align="center">
<img src="assets/gpt_kai_sve_textgen.gif" width="850" alt="Terminal recording of gpt2_kai_sve generating text at higher throughput after the SVE optimisation"/>
</p>

---

## Key Takeaways

- **Profile first, then fix.** CPU Cycle Hotspots answers "where?"; Instruction Mix answers "how?". Together they give a concrete diagnosis rather than a guess.
- **Instruction mix reveals idle hardware.** SVE = 0% on a compute-heavy workload means the processor's vector units are unused - a clear, actionable signal.
- **Always re-profile after a change.** Confirm the instruction mix shifted as expected and that throughput improved. If SVE still reads 0% after adopting an SVE library, something went wrong in the build or dispatch path.
