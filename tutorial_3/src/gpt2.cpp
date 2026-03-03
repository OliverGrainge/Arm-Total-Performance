/*
 * GPT-2 Inference in C++
 *
 * Build:
 *   g++ -O3 -march=native -fopenmp -std=c++17 -o gpt2 gpt2.cpp -lm
 *
 * Run:
 *   ./gpt2 "Once upon a time"
 *   ./gpt2 --model gpt2-medium "Once upon a time"
 *   ./gpt2 weights.bin vocab.bin "Once upon a time" -n 300 -t 0.9
 *
 * Options:
 *   -n  max new tokens (default 200)
 *   -t  temperature    (default 1.0,  0 = greedy)
 *   -p  top-p          (default 0.9)
 */

 #include <algorithm>
 #include <chrono>
 #include <climits>
 #include <cmath>
 #include <cstdint>
 #include <cstdlib>
 #include <fstream>
 #include <iostream>
 #include <numeric>
 #include <random>
 #include <string>
 #include <unordered_map>
 #include <vector>

#ifndef GPT2_DEFAULT_MODELS_DIR
#define GPT2_DEFAULT_MODELS_DIR "models"
#endif
 
 // ── helpers ──────────────────────────────────────────────────────────────────
 
 static void read_exact(std::ifstream &f, void *dst, size_t n) {
     f.read(reinterpret_cast<char *>(dst), n);
     if (!f) { std::cerr << "Unexpected EOF\n"; std::exit(1); }
 }
 
 // GPT-2 uses the tanh-based GeLU approximation
 static inline float gelu(float x) {
     return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));
 }
 
 // ── config ───────────────────────────────────────────────────────────────────
 
 struct Config {
     int vocab_size, n_ctx, n_embd, n_layer, n_head;
 };
 
 // ── weights (all float32, flat vectors) ──────────────────────────────────────
 
 struct Weights {
     std::vector<float> wte, wpe;                       // embeddings
     std::vector<float> ln1_w, ln1_b;                   // (n_layer, n_embd)
     std::vector<float> c_attn_w, c_attn_b;             // (n_layer, 3E, E) / (n_layer, 3E)
     std::vector<float> c_proj_w, c_proj_b;             // (n_layer, E, E)  / (n_layer, E)
     std::vector<float> ln2_w, ln2_b;
     std::vector<float> mlp_fc_w, mlp_fc_b;             // (n_layer, 4E, E)
     std::vector<float> mlp_pj_w, mlp_pj_b;             // (n_layer, E, 4E)
     std::vector<float> ln_f_w, ln_f_b;
 };

 
 // ── run-time state ────────────────────────────────────────────────────────────

 struct State {
     std::vector<float> x, xb, qkv, attn_out, mlp_h, logits, proj_buf;
     std::vector<float> key_cache, val_cache;   // (n_layer, n_ctx, n_embd)
     std::vector<float> att_score;              // (n_head, n_ctx)

     void init(const Config &c) {
         int E = c.n_embd;
         x.assign(E, 0); xb.assign(E, 0);
         qkv.assign(3*E, 0); attn_out.assign(E, 0);
         mlp_h.assign(4*E, 0);
         proj_buf.assign(4*E, 0);   // reusable projection scratch buffer (max dim = 4E)
         logits.assign(c.vocab_size, 0);
         key_cache.assign((size_t)c.n_layer * c.n_ctx * E, 0);
         val_cache.assign((size_t)c.n_layer * c.n_ctx * E, 0);
         att_score.assign((size_t)c.n_head  * c.n_ctx,    0);
     }
 };
 
 // ── math primitives ──────────────────────────────────────────────────────────
 
 static void layernorm(float *o, const float *x, const float *w, const float *b, int n) {
     double mean = 0, var = 0;
     for (int i = 0; i < n; i++) mean += x[i];
     mean /= n;
     for (int i = 0; i < n; i++) { double d = x[i]-mean; var += d*d; }
     float inv = 1.f / sqrtf((float)(var/n + 1e-5));
     for (int i = 0; i < n; i++) o[i] = w[i] * ((x[i]-(float)mean)*inv) + b[i];
 }
 
 // W is (n_out x n_in) row-major
 static void matmul(float *out, const float *x, const float *W, const float *b,
                    int n_in, int n_out) {
     for (int i = 0; i < n_out; i++) {
         float acc = b ? b[i] : 0.f;
         const float *row = W + (size_t)i * n_in;
         for (int j = 0; j < n_in; j++) acc += row[j] * x[j];
         out[i] = acc;
     }
 }
 
 // ── forward pass ─────────────────────────────────────────────────────────────
 
 static float *forward(int token, int pos,
                        const Config &cfg, const Weights &w, State &s)
 {
     const int E = cfg.n_embd, H = cfg.n_head, hs = E/H;
 
     // 1. Embedding
     const float *te = w.wte.data() + (size_t)token*E;
     const float *pe = w.wpe.data() + (size_t)pos  *E;
     for (int i = 0; i < E; i++) s.x[i] = te[i] + pe[i];
 
     // 2. Layers
     for (int l = 0; l < cfg.n_layer; l++) {
         // ── Attention ─────────────────────────────────────────────────────
         layernorm(s.xb.data(), s.x.data(),
                   w.ln1_w.data()+(size_t)l*E, w.ln1_b.data()+(size_t)l*E, E);
 
         matmul(s.qkv.data(), s.xb.data(),
                w.c_attn_w.data()+(size_t)l*3*E*E,
                w.c_attn_b.data()+(size_t)l*3*E,  E, 3*E);
 
         float *Q = s.qkv.data(), *K = Q+E, *V = K+E;
 
         // Cache K, V
         size_t loff = (size_t)l*cfg.n_ctx*E;
         std::copy(K, K+E, s.key_cache.data()+loff+(size_t)pos*E);
         std::copy(V, V+E, s.val_cache.data()+loff+(size_t)pos*E);
 
         std::fill(s.attn_out.begin(), s.attn_out.end(), 0.f);
         float scale = 1.f / sqrtf((float)hs);
 
         #pragma omp parallel for schedule(static)
         for (int h = 0; h < H; h++) {
             // Pointers for this head's slice of Q, and its output slot in att_score
             const float *q  = Q + h*hs;                           // this head's query vector (hs elements)
             float *sc       = s.att_score.data() + h*cfg.n_ctx;   // this head's attention scores [0..pos]
             const float *kc = s.key_cache.data() + loff;          // this layer's cached keys (all positions)
         
             // ── Step 1: Compute attention scores (Q·K^T / sqrt(hs)) ──
             for (int t = 0; t <= pos; t++) {
                 float dot = 0;
                 const float *k_t = kc + (size_t)t*E + h*hs;   // key at position t, this head's slice
                 for (int i = 0; i < hs; i++) dot += q[i]*k_t[i];
                 sc[t] = dot * scale;                          // scaled dot product → raw attention score
             }
         
             // ── Step 2: Softmax over all positions ──
             float mx = *std::max_element(sc, sc+pos+1), sm = 0;
             for (int t = 0; t<=pos; t++) { sc[t]=expf(sc[t]-mx); sm+=sc[t]; }  // subtract max for stability
             for (int t = 0; t<=pos; t++) sc[t] /= sm;                           // normalize to sum to 1
         
             // ── Step 3: Weighted sum of values → attention output for this head ──
             float *oh      = s.attn_out.data() + h*hs;       // this head's slice of the output
             const float *vc = s.val_cache.data() + loff;     // this layer's cached values (all positions)
             for (int t = 0; t <= pos; t++) {
                 const float *v_t = vc + (size_t)t*E + h*hs;  // value at position t, this head's slice
                 float a = sc[t];                              // softmax weight for position t
                 for (int i = 0; i < hs; i++) oh[i] += a*v_t[i];  // accumulate: output += a * V_t
             }
         }
 
         // Output projection + residual
         matmul(s.proj_buf.data(), s.attn_out.data(),
                w.c_proj_w.data()+(size_t)l*E*E,
                w.c_proj_b.data()+(size_t)l*E, E, E);
         for (int i=0;i<E;i++) s.x[i]+=s.proj_buf[i];
 
         // ── FFN ───────────────────────────────────────────────────────────
         layernorm(s.xb.data(), s.x.data(),
                   w.ln2_w.data()+(size_t)l*E, w.ln2_b.data()+(size_t)l*E, E);
 
         matmul(s.mlp_h.data(), s.xb.data(),
                w.mlp_fc_w.data()+(size_t)l*4*E*E,
                w.mlp_fc_b.data()+(size_t)l*4*E, E, 4*E);
         for (int i=0;i<4*E;i++) s.mlp_h[i]=gelu(s.mlp_h[i]);
 
         matmul(s.proj_buf.data(), s.mlp_h.data(),
                w.mlp_pj_w.data()+(size_t)l*E*4*E,
                w.mlp_pj_b.data()+(size_t)l*E, 4*E, E);
         for (int i=0;i<E;i++) s.x[i]+=s.proj_buf[i];
     }
 
     // 3. Final layer norm
     layernorm(s.x.data(), s.x.data(), w.ln_f_w.data(), w.ln_f_b.data(), E);
 
     // 4. Logits via weight tying  (vocab_size x n_embd) @ x
     matmul(s.logits.data(), s.x.data(), w.wte.data(), nullptr, E, cfg.vocab_size);
     return s.logits.data();
 }
 
 // ── weight loading ────────────────────────────────────────────────────────────
 
 static std::vector<float> read_tensor(std::ifstream &f, const char *name) {
     uint32_t nd; read_exact(f, &nd, 4);
     size_t total = 1;
     for (uint32_t d=0;d<nd;d++) {
         uint32_t dim; read_exact(f,&dim,4); total*=dim;
     }
     std::vector<float> v(total);
     read_exact(f, v.data(), total*4);
     std::cout << "  loaded " << name << " (" << total << ")\n";
     return v;
 }
 
 static void load_weights(const std::string &path, Config &cfg, Weights &w) {
     std::ifstream f(path, std::ios::binary);
     if (!f) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
     uint32_t magic, ver;
     read_exact(f,&magic,4); read_exact(f,&ver,4);
     if (magic != 0x67707432u) { std::cerr << "Bad magic\n"; std::exit(1); }
     read_exact(f,&cfg.vocab_size,4); read_exact(f,&cfg.n_ctx,4);
     read_exact(f,&cfg.n_embd,4);    read_exact(f,&cfg.n_layer,4);
     read_exact(f,&cfg.n_head,4);
     std::cout << "GPT-2  embd=" << cfg.n_embd << "  layers=" << cfg.n_layer
               << "  heads=" << cfg.n_head << "  vocab=" << cfg.vocab_size << "\n";
     w.wte      = read_tensor(f,"wte");
     w.wpe      = read_tensor(f,"wpe");
     w.ln1_w    = read_tensor(f,"ln1_w");
     w.ln1_b    = read_tensor(f,"ln1_b");
     w.c_attn_w = read_tensor(f,"c_attn_w");
     w.c_attn_b = read_tensor(f,"c_attn_b");
     w.c_proj_w = read_tensor(f,"c_proj_w");
     w.c_proj_b = read_tensor(f,"c_proj_b");
     w.ln2_w    = read_tensor(f,"ln2_w");
     w.ln2_b    = read_tensor(f,"ln2_b");
     w.mlp_fc_w = read_tensor(f,"mlp_fc_w");
     w.mlp_fc_b = read_tensor(f,"mlp_fc_b");
     w.mlp_pj_w = read_tensor(f,"mlp_pj_w");
     w.mlp_pj_b = read_tensor(f,"mlp_pj_b");
     w.ln_f_w   = read_tensor(f,"ln_f_w");
     w.ln_f_b   = read_tensor(f,"ln_f_b");
 }
 
 // ── tokeniser ─────────────────────────────────────────────────────────────────
 
 struct Tokenizer {
     std::vector<std::string> id2tok;
     std::unordered_map<std::string,int> tok2id;
     std::vector<std::pair<int,int>> merges;
     std::unordered_map<uint64_t,int> merge_rank;
 
     void load(const std::string &path) {
         std::ifstream f(path, std::ios::binary);
         if (!f) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }
         uint32_t magic, vsz;
         read_exact(f,&magic,4);
         if (magic != 0x62706532u) { std::cerr << "Bad vocab magic\n"; std::exit(1); }
         read_exact(f,&vsz,4);
         id2tok.resize(vsz);
         for (uint32_t i=0;i<vsz;i++) {
             uint32_t len; read_exact(f,&len,4);
             std::string tok(len,'\0'); read_exact(f,tok.data(),len);
             id2tok[i]=tok; tok2id[tok]=(int)i;
         }
         uint32_t nm; read_exact(f,&nm,4);
         merges.resize(nm);
         for (uint32_t i=0;i<nm;i++) {
             uint32_t a,b; read_exact(f,&a,4); read_exact(f,&b,4);
             merges[i]={a,b};
             merge_rank[((uint64_t)a<<32)|b]=(int)i;
         }
         std::cout << "Tokeniser: " << vsz << " tokens, " << nm << " merges\n";
     }
 
     std::vector<int> encode(const std::string &text) const {
         // Greedy longest-match seed
         std::vector<int> ids;
         for (size_t i=0; i<text.size();) {
             int best=-1; size_t best_l=0;
             for (size_t l=std::min(text.size()-i,(size_t)64);l>=1;l--) {
                 auto it=tok2id.find(text.substr(i,l));
                 if (it!=tok2id.end()) { best=it->second; best_l=l; break; }
             }
             if (best==-1) { best=0; best_l=1; }
             ids.push_back(best); i+=best_l;
         }
         // BPE merges
         while (ids.size()>=2) {
             int best_rank=INT_MAX, best_pos=-1;
             for (int j=0;j+1<(int)ids.size();j++) {
                 auto it=merge_rank.find(((uint64_t)ids[j]<<32)|(uint32_t)ids[j+1]);
                 if (it!=merge_rank.end()&&it->second<best_rank)
                     { best_rank=it->second; best_pos=j; }
             }
             if (best_pos==-1) break;
             auto &m=merges[best_rank];
             std::string merged=id2tok[m.first]+id2tok[m.second];
             auto it=tok2id.find(merged);
             ids[best_pos]=(it!=tok2id.end())?it->second:ids[best_pos];
             ids.erase(ids.begin()+best_pos+1);
         }
         return ids;
     }
     std::string piece(int id) const {
         return (id>=0&&id<(int)id2tok.size()) ? id2tok[id] : "";
     }
 };
 
 // ── sampling ─────────────────────────────────────────────────────────────────
 
 static int argmax(const float *x, int n) {
     return (int)(std::max_element(x,x+n)-x);
 }
 static int sample_topp(const float *logits, int n, float temp, float topp,
                         std::mt19937 &rng) {
     std::vector<float> p(logits,logits+n);
     for (auto &v:p) v/=temp;
     float mx=*std::max_element(p.begin(),p.end()), s=0;
     for (auto &v:p) { v=expf(v-mx); s+=v; }
     for (auto &v:p) v/=s;
     std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
     std::sort(idx.begin(),idx.end(),[&](int a,int b){return p[a]>p[b];});
     float cum=0; int cut=n;
     for (int i=0;i<n;i++) { cum+=p[idx[i]]; if (cum>=topp){cut=i+1;break;} }
     std::vector<float> w(cut);
     for (int i=0;i<cut;i++) w[i]=p[idx[i]];
     std::discrete_distribution<int> dist(w.begin(),w.end());
     return idx[dist(rng)];
 }
 
 // ── generation ────────────────────────────────────────────────────────────────
 
static void generate(const std::string &prompt, int max_new,
                     float temp, float topp,
                     const Config &cfg, const Weights &weights,
                     const Tokenizer &tok, State &state)
 {
     std::mt19937 rng(std::random_device{}());
     auto tokens = tok.encode(prompt);
     std::cout << "\n[" << tokens.size() << " prompt tokens]\n" << prompt;
 
     auto t0 = std::chrono::high_resolution_clock::now();
     int pos=0; float *logits=nullptr;
     for (int t : tokens) { logits=forward(t,pos,cfg,weights,state); pos++; }
 
     int gen=0;
     for (int step=0; step<max_new; step++) {
         int next = (temp==0.f) ? argmax(logits,cfg.vocab_size)
                                : sample_topp(logits,cfg.vocab_size,temp,topp,rng);
         if (next==50256) break;                  // <|endoftext|>
         std::cout << tok.piece(next) << std::flush;
         logits=forward(next,pos,cfg,weights,state);
         pos++; gen++;
         if (pos>=cfg.n_ctx) break;
     }
     double secs = std::chrono::duration<double>(
         std::chrono::high_resolution_clock::now()-t0).count();
    std::cout << "\n\n[" << gen << " tokens, " << gen/secs << " tok/s]\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

static std::string default_model_path(const std::string &model, const std::string &file) {
    return std::string(GPT2_DEFAULT_MODELS_DIR) + "/" + model + "/" + file;
}

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s [--model NAME] [--weights PATH --vocab PATH] [prompt] [-n N] [-t T] [-p P]\n"
        "   or: %s weights.bin vocab.bin [prompt] [-n N] [-t T] [-p P]\n", p, p);
    std::exit(1);
}

int main(int argc, char **argv) {
    std::string model = "gpt2";
    std::string wp = default_model_path(model, "weights.bin");
    std::string vp = default_model_path(model, "vocab.bin");
    std::string prompt = "Once upon a time";
    int max_new = 200;
    float temp = 1.0f, topp = 0.9f;

    int i = 1;
    if (argc >= 3 && argv[1][0] != '-' && argv[2][0] != '-') {
        wp = argv[1];
        vp = argv[2];
        i = 3;
        if (i < argc && argv[i][0] != '-') {
            prompt = argv[i++];
        }
    }

    for (; i < argc; ++i) {
        std::string f = argv[i];
        if (f == "--model") {
            if (++i >= argc) usage(argv[0]);
            model = argv[i];
            wp = default_model_path(model, "weights.bin");
            vp = default_model_path(model, "vocab.bin");
        } else if (f == "--weights") {
            if (++i >= argc) usage(argv[0]);
            wp = argv[i];
        } else if (f == "--vocab") {
            if (++i >= argc) usage(argv[0]);
            vp = argv[i];
        } else if (f == "-n") {
            if (++i >= argc) usage(argv[0]);
            max_new = std::stoi(argv[i]);
        } else if (f == "-t") {
            if (++i >= argc) usage(argv[0]);
            temp = std::stof(argv[i]);
        } else if (f == "-p") {
            if (++i >= argc) usage(argv[0]);
            topp = std::stof(argv[i]);
        } else if (!f.empty() && f[0] != '-') {
            prompt = f;
        } else {
            usage(argv[0]);
        }
    }

    Config cfg; Weights weights;
    std::cout << "Weights path: " << wp << "\n";
    std::cout << "Vocab path: " << vp << "\n";
    load_weights(wp, cfg, weights);
    Tokenizer tok; tok.load(vp);
    State state; state.init(cfg);
    generate(prompt, max_new, temp, topp, cfg, weights, tok, state);
}
