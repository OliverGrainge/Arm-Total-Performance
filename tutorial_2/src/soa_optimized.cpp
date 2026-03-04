#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// Structure-of-Arrays layout.
// The hot position-update loop only touches the x, y, z, vx, vy, vz arrays.
// Working set for those 6 arrays = 6 * 4 MB = 24 MB — fits in L3 on Graviton3.
// Every byte loaded from those arrays is useful data: 100% cache line utilisation.
struct ParticlesSoA {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    // Remaining fields exist but live in separate allocations and are never
    // touched by update_positions, so they do not pollute the hot cache lines.
    std::vector<float> mass, charge, temperature;
    std::vector<float> pressure, energy, density;
    std::vector<float> spin_x, spin_y, spin_z;
};

static void update_positions(ParticlesSoA& p, int n, float dt) {
    for (int i = 0; i < n; ++i) {
        p.x[i] += p.vx[i] * dt;
        p.y[i] += p.vy[i] * dt;
        p.z[i] += p.vz[i] * dt;
    }
}

// ----------------------------------------------------------------------------
// Minimal LCG for reproducible, dependency-free galaxy initialisation.
// Not used in the hot loop — only called once during setup.
// ----------------------------------------------------------------------------
static unsigned int lcg_state = 0x12345678u;

static float lcg_float() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float)(lcg_state >> 8) * (1.0f / 16777216.0f);
}

static float lcg_gauss() {
    float u = lcg_float() + 1e-7f;
    float v = lcg_float();
    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * 3.14159265f * v);
}

// Initialise particles as a two-arm logarithmic spiral galaxy.
// Identical initial conditions to aos_baseline — only the data layout differs.
static void init_galaxy(ParticlesSoA& p, int n) {
    const float PI      = 3.14159265f;
    const float v0      = 2.0f;
    const float winding = 3.5f;
    const float r_min   = 0.5f;
    const float r_scale = 2.2f;
    const float r_max   = 9.0f;
    const float scatter = 0.30f;
    const float z_scale = 0.15f;

    for (int i = 0; i < n; ++i) {
        float arm_offset = (i % 4) * (PI / 2.0f);

        float r = r_min - r_scale * logf(lcg_float() + 1e-7f);
        if (r > r_max) r = r_min + (r_max - r_min) * lcg_float();

        float theta = arm_offset + winding * logf(r / r_min) + lcg_gauss() * scatter;

        p.x[i]  =  r * cosf(theta);
        p.y[i]  =  r * sinf(theta);
        p.z[i]  =  lcg_gauss() * z_scale;

        p.vx[i] = -v0 * sinf(theta);
        p.vy[i] =  v0 * cosf(theta);
        p.vz[i] =  0.0f;

        p.mass[i]        = 1.0f;
        p.charge[i]      = 0.5f;
        p.temperature[i] = 300.0f;
        p.pressure[i]    = 101325.0f;
        p.energy[i]      = 0.0f;
        p.density[i]     = 1.0f;
        p.spin_x[i]      = 0.0f;
        p.spin_y[i]      = 0.0f;
        p.spin_z[i]      = 0.0f;
    }
}

int main(int argc, char* argv[]) {
    const int   N              = 1 << 20; // 1,048,576 particles — same as AoS baseline
    const int   default_iters  = 200;
    const int   vis_iters      = 1000;
    const float dt    = 0.005f;

    bool do_vis = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--visualize") == 0) {
            do_vis = true;
            break;
        }
    }

    const int iters = do_vis ? vis_iters : default_iters;

    const int vis_stride   = 16;
    const int vis_interval = 10;
    const int vis_n        = N / vis_stride;
    const int vis_frames   = 1 + iters / vis_interval;

    ParticlesSoA particles;
    particles.x.resize(N);           particles.y.resize(N);
    particles.z.resize(N);           particles.vx.resize(N);
    particles.vy.resize(N);          particles.vz.resize(N);
    particles.mass.resize(N);        particles.charge.resize(N);
    particles.temperature.resize(N); particles.pressure.resize(N);
    particles.energy.resize(N);      particles.density.resize(N);
    particles.spin_x.resize(N);      particles.spin_y.resize(N);
    particles.spin_z.resize(N);

    init_galaxy(particles, N);

    FILE* vis_fp = nullptr;
    if (do_vis) {
        vis_fp = fopen("galaxy_soa.bin", "wb");
        fwrite(&vis_n,      sizeof(int), 1, vis_fp);
        fwrite(&vis_frames, sizeof(int), 1, vis_fp);
    }

    // Helper: write one subsampled frame (x-array, then y-array, then z-array).
    auto dump_frame = [&]() {
        for (int j = 0; j < N; j += vis_stride)
            fwrite(&particles.x[j], sizeof(float), 1, vis_fp);
        for (int j = 0; j < N; j += vis_stride)
            fwrite(&particles.y[j], sizeof(float), 1, vis_fp);
        for (int j = 0; j < N; j += vis_stride)
            fwrite(&particles.z[j], sizeof(float), 1, vis_fp);
    };

    if (do_vis) dump_frame();

    for (int iter = 0; iter < iters; ++iter) {
        update_positions(particles, N, dt);

        if (do_vis && (iter + 1) % vis_interval == 0)
            dump_frame();
    }

    if (vis_fp) fclose(vis_fp);

    // Checksum — same formula as AoS baseline; values must match.
    double checksum = 0.0;
    for (int i = 0; i < N; ++i)
        checksum += particles.x[i] + particles.y[i] + particles.z[i];

    printf("SoA checksum: %.6f\n", checksum);
    return 0;
}
