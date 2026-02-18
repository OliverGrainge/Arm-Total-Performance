#include <cstdio>
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

int main() {
    const int   N     = 1 << 20; // 1,048,576 particles — working set = 64 MB
    const int   iters = 200;
    const float dt    = 0.001f;

    std::vector<ParticleAoS> particles(N);

    for (int i = 0; i < N; ++i) {
        particles[i].x           = (float)i * 0.1f;
        particles[i].y           = (float)i * 0.2f;
        particles[i].z           = (float)i * 0.3f;
        particles[i].vx          = 1.0f;
        particles[i].vy          = 2.0f;
        particles[i].vz          = 3.0f;
        particles[i].mass        = 1.0f;
        particles[i].charge      = 0.5f;
        particles[i].temperature = 300.0f;
        particles[i].pressure    = 101325.0f;
        particles[i].energy      = 0.0f;
        particles[i].density     = 1.0f;
        particles[i].spin_x      = 0.0f;
        particles[i].spin_y      = 0.0f;
        particles[i].spin_z      = 0.0f;
        particles[i].pad         = 0.0f;
    }

    for (int iter = 0; iter < iters; ++iter)
        update_positions(particles.data(), N, dt);

    // Checksum — must match soa_optimized for correctness verification.
    double checksum = 0.0;
    for (int i = 0; i < N; ++i)
        checksum += particles[i].x + particles[i].y + particles[i].z;

    printf("AoS checksum: %.6f\n", checksum);
    return 0;
}
