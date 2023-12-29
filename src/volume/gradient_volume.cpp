#include "gradient_volume.h"
#include <algorithm>
#include <exception>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <gsl/span>

namespace volume {

// Compute the maximum magnitude from all gradient voxels
static float computeMaxMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::max_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute the minimum magnitude from all gradient voxels
static float computeMinMagnitude(gsl::span<const GradientVoxel> data)
{
    return std::min_element(
        std::begin(data),
        std::end(data),
        [](const GradientVoxel& lhs, const GradientVoxel& rhs) {
            return lhs.magnitude < rhs.magnitude;
        })
        ->magnitude;
}

// Compute a gradient volume from a volume
static std::vector<GradientVoxel> computeGradientVolume(const Volume& volume)
{
    const auto dim = volume.dims();

    std::vector<GradientVoxel> out(static_cast<size_t>(dim.x * dim.y * dim.z));
    for (int z = 1; z < dim.z - 1; z++) {
        for (int y = 1; y < dim.y - 1; y++) {
            for (int x = 1; x < dim.x - 1; x++) {
                const float gx = (volume.getVoxel(x + 1, y, z) - volume.getVoxel(x - 1, y, z)) / 2.0f;
                const float gy = (volume.getVoxel(x, y + 1, z) - volume.getVoxel(x, y - 1, z)) / 2.0f;
                const float gz = (volume.getVoxel(x, y, z + 1) - volume.getVoxel(x, y, z - 1)) / 2.0f;

                const glm::vec3 v { gx, gy, gz };
                const size_t index = static_cast<size_t>(x + dim.x * (y + dim.y * z));
                out[index] = GradientVoxel { v, glm::length(v) };
            }
        }
    }
    return out;
}

GradientVolume::GradientVolume(const Volume& volume)
    : m_dim(volume.dims())
    , m_data(computeGradientVolume(volume))
    , m_minMagnitude(computeMinMagnitude(m_data))
    , m_maxMagnitude(computeMaxMagnitude(m_data))
{
}

float GradientVolume::maxMagnitude() const
{
    return m_maxMagnitude;
}

float GradientVolume::minMagnitude() const
{
    return m_minMagnitude;
}

glm::ivec3 GradientVolume::dims() const
{
    return m_dim;
}

// This function returns a gradientVoxel at coord based on the current interpolation mode.
GradientVoxel GradientVolume::getGradientInterpolate(const glm::vec3& coord) const
{
    switch (interpolationMode) {
    case InterpolationMode::NearestNeighbour: {
        return getGradientNearestNeighbor(coord);
    }
    case InterpolationMode::Linear: {
        return getGradientLinearInterpolate(coord);
    }
    case InterpolationMode::Cubic: {
        // No cubic in this case, linear is good enough for the gradient.
        return getGradientLinearInterpolate(coord);
    }
    default: {
        throw std::exception();
    }
    };
}

// This function returns the nearest neighbour given a position in the volume given by coord.
// Notice that in this framework we assume that the distance between neighbouring voxels is 1 in all directions
GradientVoxel GradientVolume::getGradientNearestNeighbor(const glm::vec3& coord) const
{
    if (glm::any(glm::lessThan(coord, glm::vec3(0))) || glm::any(glm::greaterThanEqual(coord, glm::vec3(m_dim))))
        return { glm::vec3(0.0f), 0.0f };

    auto roundToPositiveInt = [](float f) {
        return static_cast<int>(f + 0.5f);
    };

    return getGradient(roundToPositiveInt(coord.x), roundToPositiveInt(coord.y), roundToPositiveInt(coord.z));
}

// ======= TODO : IMPLEMENT ========
// Returns the trilinearly interpolated gradinet at the given coordinate.
// Use the linearInterpolate function that you implemented below.
GradientVoxel GradientVolume::getGradientLinearInterpolate(const glm::vec3& coord) const
{
    //get the 8 points around the coord
    int x0 = static_cast<int>(floor(coord.x));
    int x1 = static_cast<int>(ceil(coord.x));
    int y0 = static_cast<int>(floor(coord.y));
    int y1 = static_cast<int>(ceil(coord.y));
    int z0 = static_cast<int>(floor(coord.z));
    int z1 = static_cast<int>(ceil(coord.z));

    //get the 8 gradients
    GradientVoxel g000 = getGradient(x0, y0, z0);
    GradientVoxel g001 = getGradient(x0, y0, z1);
    GradientVoxel g010 = getGradient(x0, y1, z0);
    GradientVoxel g011 = getGradient(x0, y1, z1);
    GradientVoxel g100 = getGradient(x1, y0, z0);
    GradientVoxel g101 = getGradient(x1, y0, z1);
    GradientVoxel g110 = getGradient(x1, y1, z0);
    GradientVoxel g111 = getGradient(x1, y1, z1);

    //get the 8 magnitudes
    float m000 = g000.magnitude;
    float m001 = g001.magnitude;
    float m010 = g010.magnitude;
    float m011 = g011.magnitude;

    float m100 = g100.magnitude;
    float m101 = g101.magnitude;
    float m110 = g110.magnitude;
    float m111 = g111.magnitude;

    //get the 8 directions
    glm::vec3 d000 = g000.dir;
    glm::vec3 d001 = g001.dir;
    glm::vec3 d010 = g010.dir;
    glm::vec3 d011 = g011.dir;

    glm::vec3 d100 = g100.dir;
    glm::vec3 d101 = g101.dir;
    glm::vec3 d110 = g110.dir;
    glm::vec3 d111 = g111.dir;

    //get the 8 points
    glm::vec3 p000 = glm::vec3(x0, y0, z0);
    glm::vec3 p001 = glm::vec3(x0, y0, z1);
    glm::vec3 p010 = glm::vec3(x0, y1, z0);
    glm::vec3 p011 = glm::vec3(x0, y1, z1);

    glm::vec3 p100 = glm::vec3(x1, y0, z0);
    glm::vec3 p101 = glm::vec3(x1, y0, z1);
    glm::vec3 p110 = glm::vec3(x1, y1, z0);
    glm::vec3 p111 = glm::vec3(x1, y1, z1);

    //get the 8 distances
    float d000_ = glm::distance(coord, p000);
    float d001_ = glm::distance(coord, p001);
    float d010_ = glm::distance(coord, p010);
    float d011_ = glm::distance(coord, p011);

    float d100_ = glm::distance(coord, p100);
    float d101_ = glm::distance(coord, p101);
    float d110_ = glm::distance(coord, p110);
    float d111_ = glm::distance(coord, p111);

    //get the 8 factors
    float f000 = 1 / d000_;
    float f001 = 1 / d001_;
    float f010 = 1 / d010_;
    float f011 = 1 / d011_;

    float f100 = 1 / d100_;
    float f101 = 1 / d101_;
    float f110 = 1 / d110_;
    float f111 = 1 / d111_;

    //get the 8 gradients
    GradientVoxel g000_ = GradientVoxel { d000, m000 };
    GradientVoxel g001_ = GradientVoxel { d001, m001 };
    GradientVoxel g010_ = GradientVoxel { d010, m010 };
    GradientVoxel g011_ = GradientVoxel { d011, m011 };

    GradientVoxel g100_ = GradientVoxel { d100, m100 };
    GradientVoxel g101_ = GradientVoxel { d101, m101 };
    GradientVoxel g110_ = GradientVoxel { d110, m110 };
    GradientVoxel g111_ = GradientVoxel { d111, m111 };

    //get the 4 gradients
    GradientVoxel g00 = linearInterpolate(g000_, g001_, f000);
    GradientVoxel g01 = linearInterpolate(g010_, g011_, f001);
    GradientVoxel g10 = linearInterpolate(g100_, g101_, f010);
    GradientVoxel g11 = linearInterpolate(g110_, g111_, f011);

    //get the 2 gradients
    GradientVoxel g0 = linearInterpolate(g00, g01, f000);
    GradientVoxel g1 = linearInterpolate(g10, g11, f001);

    //get the 1 gradient
    GradientVoxel g = linearInterpolate(g0, g1, f000);

    return g;
}

// ======= TODO : IMPLEMENT ========
// This function should linearly interpolates the value from g0 to g1 given the factor (t).
// At t=0, linearInterpolate should return g0 and at t=1 it returns g1.
GradientVoxel GradientVolume::linearInterpolate(const GradientVoxel& g0, const GradientVoxel& g1, float factor)
{
    //linear interpolation of the magnitude
    float magnitude = g0.magnitude + factor * (g1.magnitude - g0.magnitude);

    //linear interpolation of the direction
    glm::vec3 direction = g0.dir + factor * (g1.dir - g0.dir);

    return GradientVoxel { direction, magnitude };
}

// This function returns a gradientVoxel without using interpolation
GradientVoxel GradientVolume::getGradient(int x, int y, int z) const
{
    const size_t i = static_cast<size_t>(x + m_dim.x * (y + m_dim.y * z));
    return m_data[i];
}
}