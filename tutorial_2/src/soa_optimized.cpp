#include <cstdio>
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

int main() {
    const int   N     = 1 << 20; // 1,048,576 particles — same as AoS baseline
    const int   iters = 200;
    const float dt    = 0.001f;

    ParticlesSoA particles;
    particles.x.resize(N);           particles.y.resize(N);
    particles.z.resize(N);           particles.vx.resize(N);
    particles.vy.resize(N);          particles.vz.resize(N);
    particles.mass.resize(N);        particles.charge.resize(N);
    particles.temperature.resize(N); particles.pressure.resize(N);
    particles.energy.resize(N);      particles.density.resize(N);
    particles.spin_x.resize(N);      particles.spin_y.resize(N);
    particles.spin_z.resize(N);

    for (int i = 0; i < N; ++i) {
        particles.x[i]           = (float)i * 0.1f;
        particles.y[i]           = (float)i * 0.2f;
        particles.z[i]           = (float)i * 0.3f;
        particles.vx[i]          = 1.0f;
        particles.vy[i]          = 2.0f;
        particles.vz[i]          = 3.0f;
        particles.mass[i]        = 1.0f;
        particles.charge[i]      = 0.5f;
        particles.temperature[i] = 300.0f;
        particles.pressure[i]    = 101325.0f;
        particles.energy[i]      = 0.0f;
        particles.density[i]     = 1.0f;
        particles.spin_x[i]      = 0.0f;
        particles.spin_y[i]      = 0.0f;
        particles.spin_z[i]      = 0.0f;
    }

    for (int iter = 0; iter < iters; ++iter)
        update_positions(particles, N, dt);

    // Checksum — same formula as AoS baseline; values must match.
    double checksum = 0.0;
    for (int i = 0; i < N; ++i)
        checksum += particles.x[i] + particles.y[i] + particles.z[i];

    printf("SoA checksum: %.6f\n", checksum);
    return 0;
}
