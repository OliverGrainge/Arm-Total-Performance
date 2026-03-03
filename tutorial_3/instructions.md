# Tutorial 3: GPT-2 with KleidiAI

## 1. AWS instance setup (first-time)

On your Graviton instance (Amazon Linux):

```bash
sudo dnf install -y git cmake gcc-c++ libgomp python3 python3-pip
```

Clone the repo:

```bash
git clone --recurse-submodules https://github.com/OliverGrainge/Arm-Total-Performance.git
cd Arm-Total-Performance/tutorial_3
```

## 2. Export weights (requires Python)

Install Python dependencies:

```bash
pip3 install torch transformers
```

Export GPT-2 weights and vocab:

```bash
python3 src/export_gpt2.py
python3 src/export_gpt2.py --model gpt2-medium
python3 src/export_gpt2.py --model gpt2-large
python3 src/export_gpt2.py --model gpt2-xl
```

Weights are written to `models/gpt2/weights.bin` and `models/gpt2/vocab.bin` (and `models/gpt2-medium/` for the medium model).

## 3. Build

```bash
cmake -S . -B build
cmake --build build --parallel
```

## 4. Run

From the `build` directory:

```bash
cd build
./gpt2 "Once upon a time"
./gpt2_kleidiai "Once upon a time"
```

For GPT-2 medium:

```bash
./gpt2 --model gpt2-medium "Once upon a time"
```

### Options

- `-n N`  max new tokens (default 200)
- `-t T`  temperature (default 1.0, 0 = greedy)
- `-p P`  top-p (default 0.9)

Example:

```bash
./gpt2 "Once upon a time" -n 300 -t 0.9
```
