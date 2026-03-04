#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// Array-of-Structures layout.
// Each ParticleAoS is exactly 64 bytes — one full cache line.
// The hot position-update loop only reads/writes x, y, z, vx, vy, vz
// (6 floats = 24 bytes), so 40 of the 64 bytes loaded per particle are wasted.
struct ParticleAoS {
    float x, y, z;                   // position      (12 bytes) — used in hot loop
    float vx, vy, vz;                // velocity      (12 bytes) — used in hot loop
    float mass, charge, temperature; // properties    (12 bytes) — not used in hot loop
    float pressure, energy, density; //               (12 bytes) — not used in hot loop
    float spin_x, spin_y, spin_z;    //               (12 bytes) — not used in hot loop
    float pad;                        // padding        (4 bytes)
    // Total: 64 bytes = 1 cache line. Hot loop uses 24 / 64 = 37.5%.
};

static void update_positions(ParticleAoS* p, int n, float dt) {
    for (int i = 0; i < n; ++i) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
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
    // Box-Muller transform: two uniform samples → one Gaussian sample.
    float u = lcg_float() + 1e-7f;
    float v = lcg_float();
    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * 3.14159265f * v);
}

// Initialise particles as a two-arm logarithmic spiral galaxy.
// Particles are placed on spiral arms with a flat rotation curve so that
// differential rotation slowly winds the arms further during the simulation.
static void init_galaxy(ParticleAoS* p, int n) {
    const float PI      = 3.14159265f;
    const float v0      = 2.0f;   // orbital speed (flat rotation curve)
    const float winding = 3.5f;   // logarithmic spiral winding constant
    const float r_min   = 0.5f;   // inner edge of disk
    const float r_scale = 2.2f;   // exponential scale radius
    const float r_max   = 9.0f;   // outer cutoff
    const float scatter = 0.30f;  // angular scatter around arm centreline
    const float z_scale = 0.15f;  // disk half-thickness

    for (int i = 0; i < n; ++i) {
        // Distribute particles evenly across four arms (offset by π/2 each).
        float arm_offset = (i % 4) * (PI / 2.0f);

        // Sample radius from an exponential distribution, clamped to [r_min, r_max].
        float r = r_min - r_scale * logf(lcg_float() + 1e-7f);
        if (r > r_max) r = r_min + (r_max - r_min) * lcg_float();

        // Logarithmic spiral: θ = arm_offset + winding * ln(r / r_min) + scatter
        float theta = arm_offset + winding * logf(r / r_min) + lcg_gauss() * scatter;

        p[i].x  =  r * cosf(theta);
        p[i].y  =  r * sinf(theta);
        p[i].z  =  lcg_gauss() * z_scale;

        // Flat rotation curve: tangential speed = v0 regardless of radius.
        // vtan direction = (-sin θ, cos θ, 0).
        p[i].vx =  -v0 * sinf(theta);
        p[i].vy =   v0 * cosf(theta);
        p[i].vz =   0.0f;

        // Cold fields — present in the struct but never touched by update_positions.
        p[i].mass        = 1.0f;
        p[i].charge      = 0.5f;
        p[i].temperature = 300.0f;
        p[i].pressure    = 101325.0f;
        p[i].energy      = 0.0f;
        p[i].density     = 1.0f;
        p[i].spin_x      = 0.0f;
        p[i].spin_y      = 0.0f;
        p[i].spin_z      = 0.0f;
        p[i].pad         = 0.0f;
    }
}

int main(int argc, char* argv[]) {
    const int   N     = 1 << 20; // 1,048,576 particles — working set = 64 MB
    const int   iters = 200;
    const float dt    = 0.005f;

    // --visualize: dump subsampled position snapshots for the Python visualiser.
    // Omit this flag when profiling with ATP to avoid I/O overhead.
    const bool do_vis = (argc > 1 && strcmp(argv[1], "--visualize") == 0);

    // Subsample 1-in-16 particles for compact output (~65 k points per frame).
    const int vis_stride   = 16;
    const int vis_interval = 10;  // dump every 10 iterations
    const int vis_n        = N / vis_stride;
    const int vis_frames   = 1 + iters / vis_interval; // frame 0 + 20 evolved frames

    std::vector<ParticleAoS> particles(N);
    init_galaxy(particles.data(), N);

    FILE* vis_fp = nullptr;
    if (do_vis) {
        vis_fp = fopen("galaxy_aos.bin", "wb");
        fwrite(&vis_n,      sizeof(int), 1, vis_fp);
        fwrite(&vis_frames, sizeof(int), 1, vis_fp);
    }

    // Helper: write one subsampled frame (x-array, then y-array, then z-array).
    auto dump_frame = [&]() {
        for (int j = 0; j < N; j += vis_stride)
            fwrite(&particles[j].x, sizeof(float), 1, vis_fp);
        for (int j = 0; j < N; j += vis_stride)
            fwrite(&particles[j].y, sizeof(float), 1, vis_fp);
        for (int j = 0; j < N; j += vis_stride)
            fwrite(&particles[j].z, sizeof(float), 1, vis_fp);
    };

    // Frame 0: initial galaxy shape before any position update.
    if (do_vis) dump_frame();

    for (int iter = 0; iter < iters; ++iter) {
        update_positions(particles.data(), N, dt);

        if (do_vis && (iter + 1) % vis_interval == 0)
            dump_frame();
    }

    if (vis_fp) fclose(vis_fp);

    // Checksum — must match soa_optimized for correctness verification.
    double checksum = 0.0;
    for (int i = 0; i < N; ++i)
        checksum += particles[i].x + particles[i].y + particles[i].z;

    printf("AoS checksum: %.6f\n", checksum);
    return 0;
}
