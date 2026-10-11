#ifndef ACID_RAIN_CORE_H
#define ACID_RAIN_CORE_H

#include <stdint.h>
#include <simd/simd.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ARCore ARCore;

/* Exactly 208 bytes; also shared by the Metal shaders. Linear-light textures. */
typedef struct ARUniforms {
    vector_float4 viewport; /* drawable width, height, inverse width, inverse height: host fills */
    vector_float4 time;     /* scene seconds, wall delta, trip, RGB offset in logical pixels */
    vector_float4 camera;   /* camera target x/y in local metres [0,2], roll, elevation */
    vector_float4 optics;   /* horizontal field of view radians, absorption, exposure, saturation */
    vector_float4 gain;     /* RGB gain, contrast */
    vector_float4 light;    /* unit direction xyz, interference strength */
    vector_float4 world;    /* physical seconds, min/max surface height, scene (0..6) */
    vector_float4 post;     /* effects enabled, quality (0 low, 1 balanced, 2 high), reserved */
    vector_float4 eye;      /* host fills fitted camera position xyz */
    vector_float4 forward;  /* host fills fitted camera direction xyz */
    vector_float4 right;    /* host fills horizontal camera basis xyz, horizontal tangent w */
    vector_float4 up;       /* host fills vertical camera basis xyz, vertical tangent w */
    vector_float4 illumination; /* core supplies upper-hemisphere E/pi RGB; host W selects observation lighting */
} ARUniforms;

/* Exactly 64 bytes, 16-byte aligned; shared by Rust, Swift and Metal.
   IDs never repeat, even when clouds are cleared. State 0 is airborne, 1 is
   an impact retained for four physical seconds so skipped frames lose no source. */
typedef struct ARDrop {
    vector_float4 position; /* local x/y/z metres, visible sphere radius metres */
    vector_float4 pigment;  /* normalized pigment species, total integral c*dV */
    vector_float4 dynamics; /* source sigma XY/Z metres, impact physical time, ring speed m/s */
    vector_uint4 identity;  /* ID low/high words, state (0 air, 1 impact), reserved */
} ARDrop;

typedef struct ARSnapshot {
    uint32_t water_width, water_height, bed_width, bed_height, sky_width, sky_height;
    uint32_t light_width, light_height;
    /* Borrowed, row-major RGBA float32. Valid until the next mutating call or destroy. */
    const float *water; /* height metres, slope dx, slope dy, foam */
    const float *dye;   /* three pigment areal masses, actual water depth metres */
    /* Water/dye nodes are at local=(2*x/width,2*y/height). Sample Metal normalized
       UV=local/2 + .5/float2(width,height) to align the existing physical lattice. */
    const float *bed;   /* neutral albedo, emitted R/G/B; covers [-1.5,3.5]^2 local metres */
    /* Bed and light texel centres: local=-1.5+5*(index+.5)/dimension on each axis. */
    const float *sky;   /* RGB radiance, 1; equirectangular theta=2pi*u, z=cos(pi*v) */
    /* Sky texel centres: u=(x+.5)/width, v=(y+.5)/height; dir=(sin(pi*v)*cos(2pi*u),
       sin(pi*v)*sin(2pi*u),cos(pi*v)). Sampler repeats U, clamps V. */
    const float *bed_light; /* RGB incident light, 1; same [-1.5,3.5]^2 footprint as bed */
    ARUniforms uniforms;
    float speed, rain, drop_size, bleach;
    uint32_t mode; /* 0 classic, 1 drain, 2 rapids */
    uint32_t scene; /* 0 journey, 1 acid, 2 aurora, 3 storm, 4 abyss, 5 lava, 6 prism */
    uint32_t travel, current, paused, quality, effects;
    uint64_t frame;
    uint64_t bed_revision; /* Upload bed only when this changes. */
    double world_origin_x, world_origin_y; /* exact lower-left local window in world metres */
    double physical_time; /* volume and ballistic clock; unchanged while paused */
    const ARDrop *drops; /* complete airborne state plus retained impact journal */
    uint32_t drop_count; /* at most 512 records, ordered by increasing ID */
    uint32_t reserved;
    uint64_t cloud_revision; /* clear volume/journal whenever this generation changes */
    const float *flow; /* row-major RGBA float32: depth-averaged u/v m/s, 0, depth metres */
} ARSnapshot;

enum ARCommand {
    AR_PAUSE = 1,          /* toggle */
    AR_SPEED = 2,          /* absolute 0.1..3 */
    AR_RAIN = 3,           /* absolute 0..60 drops per physical second */
    AR_DROP_SIZE = 4,      /* absolute 0.25..3 diameter multiplier */
    AR_BLEACH = 5,         /* absolute 0..4 */
    AR_TRIP = 6,           /* absolute 0..4 */
    AR_RGB_SHIFT = 7,      /* absolute 0..12 logical pixels */
    AR_QUALITY = 8,        /* absolute 0 low, 1 balanced, 2 high */
    AR_MODE = 9,           /* absolute 0 classic, 1 drain, 2 rapids */
    AR_SCENE = 10,         /* absolute 0 journey ... 6 prism */
    AR_TRAVEL = 11,        /* toggle */
    AR_CURRENT = 12,       /* toggle */
    AR_EFFECTS = 13,       /* toggle RGB separation and retinal effects */
    AR_BURST = 14,         /* spawn 12 visible falling drops */
    AR_SINGLE_DROP = 15,   /* spawn one visible drop over the current camera target */
    AR_CLEAR_CLOUDS = 16   /* clear pending drops and volume generation; retain waves */
};

/* All calls must be serialized. No function allows Rust unwinding across C. */
ARCore *ar_core_create(uint64_t seed);
void ar_core_destroy(ARCore *core);
/* Return 1 on success, 0 for null/invalid inputs or a contained internal panic. */
int32_t ar_core_step(ARCore *core, double wall_delta_seconds);
int32_t ar_core_command(ARCore *core, uint32_t command, float value);
/* Spawn one visible drop at world-space coordinates. Size is a 0.25..3 diameter
   multiplier. Returns 0 for invalid/out-of-window coordinates or a full journal. */
int32_t ar_core_drop(ARCore *core, double world_x, double world_y, float size);
int32_t ar_core_snapshot(const ARCore *core, ARSnapshot *snapshot);
uint32_t ar_core_abi_version(void); /* currently 2 */

#ifdef __cplusplus
}
#endif
#endif
