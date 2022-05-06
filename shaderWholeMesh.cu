//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include<optix.h>
#include <optix_device.h>

#include "../optixProject.h"
#include <cuda/helpers.h>


#include <sutil/vec_math.h>

#define AIR_INDEX 1.0f

extern "C" {
    __constant__ Params params;
}


static __forceinline__ __device__ void setPayloadRadiance(RadiancePRD p)
{
    optixSetPayload_0(float_as_int(p.color.x));
    optixSetPayload_1(float_as_int(p.color.y));
    optixSetPayload_2(float_as_int(p.color.z));
    optixSetPayload_3(p.depth);
    optixSetPayload_4(float_as_int(p.colorWeight));

}

static __forceinline__ __device__ RadiancePRD getPayloadRadiance()
{
    RadiancePRD p;
    p.color.x = int_as_float(optixGetPayload_0());
    p.color.y = int_as_float(optixGetPayload_1());
    p.color.z = int_as_float(optixGetPayload_2());
    p.depth = optixGetPayload_3();
    p.colorWeight = int_as_float(optixGetPayload_4());
    return p;
}


static __forceinline__ __device__ void setPayloadOcclusion(OcclusionPRD p)
{
    optixSetPayload_0(p.isShadowed);
}

static __forceinline__ __device__ OcclusionPRD getPayloadOcclusion()
{
    OcclusionPRD p;
    p.isShadowed = optixGetPayload_0();
    return p;
}




static __forceinline__ __device__ LocalGeometry getLocalGeometry(HitGroupData* sbtData, const unsigned int prim_idx)
{
    LocalGeometry geomData;


    
    const float2 barys = optixGetTriangleBarycentrics();

    int3 vertexIndex = sbtData->vertexIndex[prim_idx];

    const float3 P0 = sbtData->vertex[vertexIndex.x];
    const float3 P1 = sbtData->vertex[vertexIndex.y];
    const float3 P2 = sbtData->vertex[vertexIndex.z];

    geomData.P = (1.0f - barys.x - barys.y) * P0 + barys.x * P1 + barys.y * P2;
    geomData.P = optixTransformPointFromObjectToWorldSpace(geomData.P);


    float2 UV0, UV1, UV2;
   
    if (sbtData->texcoord)
    {
        int3 texcoordIndex = sbtData->texcoordIndex[prim_idx];
        UV0 = sbtData->texcoord[texcoordIndex.x];
        UV1 = sbtData->texcoord[texcoordIndex.y];
        UV2 = sbtData->texcoord[texcoordIndex.z];

        geomData.UV = (1.0f - barys.x - barys.y) * UV0 + barys.x * UV1 + barys.y * UV2;
    }
    else
    {
        UV0 = make_float2(0.0f, 0.0f);
        UV1 = make_float2(0.0f, 1.0f);
        UV2 = make_float2(1.0f, 0.0f);
        geomData.UV = barys;
    }

    geomData.Ng = normalize(cross(P1 - P0, P2 - P0));
    geomData.Ng = optixTransformNormalFromObjectToWorldSpace(geomData.Ng);


    float3 N0, N1, N2;
    if (sbtData->normal)
    {
        int3 normalIndex = sbtData->normalIndex[prim_idx];
        N0 = sbtData->normal[normalIndex.x];
        N1 = sbtData->normal[normalIndex.y];
        N2 = sbtData->normal[normalIndex.z];
        geomData.N = (1.0f - barys.x - barys.y) * N0 + barys.x * N1 + barys.y * N2;
        geomData.N = normalize(optixTransformNormalFromObjectToWorldSpace(geomData.N));
    }
    else
    {
        geomData.N = N0 = N1 = N2 = geomData.Ng;
    }

    /*
    const float du1 = UV0.x - UV2.x;
    const float du2 = UV1.x - UV2.x;
    const float dv1 = UV0.y - UV2.y;
    const float dv2 = UV1.y - UV2.y;

    const float3 dp1 = P0 - P2;
    const float3 dp2 = P1 - P2;

    const float3 dn1 = N0 - N2;
    const float3 dn2 = N1 - N2;

    const float det = du1 * dv2 - dv1 * du2;

    const float invdet = 1.f / det;
    geomData.dpdu = (dv2 * dp1 - dv1 * dp2) * invdet;
    geomData.dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
    geomData.dndu = (dv2 * dn1 - dv1 * dn2) * invdet;
    geomData.dndv = (-du2 * dn1 + du1 * dn2) * invdet;
    */
    return geomData;
}


static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
    /*
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)
    ) - 1.0f;

    origin = params.eye;
    direction = normalize(d.x * U + d.y * V + W);
    */

    //modify start
    origin = params.eye;
    direction = normalize((params.startPoint + params.stepX * (static_cast<float>(idx.x) + 0.5f) * params.U - params.stepY * (static_cast<float>(idx.y) + 0.5f) * params.V) - params.eye);
    //modify end
}

static __forceinline__ __device__ float3 getReflactionRayDirection(float3 pointToRayOrigin, float3 pointNormal, float refractionIndex)
{
    float ddotn, ddotn2, n_div_nt, n_div_nt2;
    float sqrt_part;
    float in_sqrt;
    float3 nextDir;
    pointToRayOrigin = -1.0f * pointToRayOrigin;
    if (refractionIndex == AIR_INDEX) {
        nextDir = pointToRayOrigin;
        return nextDir;
    }
    ddotn = dot(pointNormal, pointToRayOrigin);
    if (ddotn == 0.) {
        nextDir = pointToRayOrigin;
        return nextDir;
    }
    if (ddotn < 0.) {  // normal case (from air to obj.)
        n_div_nt = AIR_INDEX / refractionIndex;
    }
    else {  // (from obj. to air)
        n_div_nt = refractionIndex / AIR_INDEX;
    }
    ddotn2 = ddotn * ddotn;
    n_div_nt2 = n_div_nt * n_div_nt;
    in_sqrt = 1.0f - n_div_nt2 * (1.0f - ddotn2);
    in_sqrt = fabs(in_sqrt);
    sqrt_part = static_cast<float>(sqrt(in_sqrt));
    if (n_div_nt < 1.0) {
        nextDir = n_div_nt * (pointToRayOrigin - pointNormal * ddotn) - pointNormal * sqrt_part;
    }
    else {
        nextDir = n_div_nt * (pointToRayOrigin - pointNormal * ddotn) + pointNormal * sqrt_part;
    }
    return normalize(nextDir);

}

static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle      handle,
    float3                      ray_origin,
    float3                      ray_direction,
    float                       tmin,
    float                       tmax,
    RadiancePRD* payload
)
{
    unsigned int u0 = 0, u1 = 0, u2 = 0, u3 = 0, u4 = 0;
    u3 = payload->depth;
    u4 = float_as_int(payload->colorWeight);
    optixTrace(
        handle,
        ray_origin, ray_direction,
        tmin,
        tmax,
        0.0f,                     // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,       // SBT offset
        RAY_TYPE_COUNT,           // SBT stride
        RAY_TYPE_RADIANCE,        // missSBTIndex
        u0, u1, u2, u3, u4);
    payload->color.x = int_as_float(u0);
    payload->color.y = int_as_float(u1);
    payload->color.z = int_as_float(u2);
    payload->depth = u3;
    payload->colorWeight = int_as_float(u4);
}


static __forceinline__ __device__ void traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    OcclusionPRD* p
)
{
    unsigned int isShadowed = 1u;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                    // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RAY_TYPE_OCCLUSION,      // SBT offset
        RAY_TYPE_COUNT,          // SBT stride
        RAY_TYPE_OCCLUSION,      // missSBTIndex
        isShadowed);
    p->isShadowed = isShadowed;
}






extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();


    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay(idx, dim, ray_origin, ray_direction);

    // Trace the ray against our scene hierarchy

    RadiancePRD payload;
    payload.color = make_float3(0.0f);
    payload.colorWeight = 1.0f;
    payload.depth = 0;
    traceRadiance(params.handle, ray_origin, ray_direction,
        0.01f,  // tmin       
        1e16f,  // tmax
        &payload);


    // Record results in our output raster
    params.frame_buffer[(dim.y - idx.y) * params.width + idx.x] = make_color(payload.color);
}



extern "C" __global__ void __miss__radiance()
{
    MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD p = getPayloadRadiance();
    p.color = miss_data->bg_color;
    setPayloadRadiance(p);
}



extern "C" __global__ void __closesthit__radiance()
{


    HitGroupData* sbtWholeData
        = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer()); //hit point의 쉐이딩 정보


    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const LocalGeometry          geom = getLocalGeometry(sbtWholeData, prim_idx); //hit point의 기하 정보 계산.


    Material *sbtData = &(sbtWholeData->materials[sbtWholeData->materialIDs[prim_idx]]);

    float3 V = -normalize(optixGetWorldRayDirection()); // hit point to ray origin vector.
    
    
    float3 N = geom.N; //hit point의 normal.

    if (sbtData->dissolve > 0.0f && dot(V, N) < 0.0f) {
        N *= -1.0f;
    }

    RadiancePRD radiancePayload = getPayloadRadiance();

   
    float3 color = sbtData->ambient;
    
    
    
    for (int lightID = 0; lightID < params.lights.count; ++lightID) //모든 빛에 대해서
    {
        
        float3 L = params.lights[lightID].pos - geom.P; // hit point to light pos vector.
        float L_dist = length(L);
        L /= L_dist; // normalize N.
        


        float3 R = reflect(-L, N); // 빛의 반사 벡터. reflect 함수 특성상 -L 넘겨줌.

        OcclusionPRD occlusionPayload;
        
        traceOcclusion(params.handle,
            geom.P + L * 0.01f, //hit point 에서
            L, //빛 방향으로 shadow ray.
            0.001f,  // tmin       
            L_dist + 0.01f,  // tmax
            &occlusionPayload
        );
        
        
        float nDotL = fmax(0.0f, dot(N, L));
        float vDotR = fmax(0.0f, dot(V, R));
       
        color += params.lights[lightID].color * sbtData->diffuse * nDotL * occlusionPayload.isShadowed;
        
        
        color += sbtData->specular * powf(vDotR, sbtData->shininess) * occlusionPayload.isShadowed; //shininess 가 material의 specular exponent인지 확실치 않음.
        
    }
    
    if (sbtData->diffuseTextureID != -1)
    {
        float4 textureColor = tex2D<float4>(sbtWholeData->textures[sbtData->diffuseTextureID], geom.UV.x, geom.UV.y);
        color *= make_float3(textureColor);
    }
    
    color *= radiancePayload.colorWeight * (1.0f - (sbtData->metallic + sbtData->dissolve) * (radiancePayload.depth < params.maxTraceDepth));

    

    if (radiancePayload.depth < params.maxTraceDepth && sbtData->metallic>0.0f) //반사 구현
    {
        RadiancePRD reflectionRadiancePayload;
        float3 R = reflect(-V, N);

        reflectionRadiancePayload.colorWeight = radiancePayload.colorWeight * sbtData->metallic;
        reflectionRadiancePayload.depth = radiancePayload.depth + 1;
        traceRadiance(params.handle,
            geom.P + R * 0.01f,
            R,
            0.001f,
            1e16f,
            &reflectionRadiancePayload
        );
        color += reflectionRadiancePayload.color;
    }
    if (radiancePayload.depth < params.maxTraceDepth && sbtData->dissolve > 0.0f) //굴절 구현
    {
        RadiancePRD refractionRadiancePayload;
        float3 R = getReflactionRayDirection(V, N, sbtData->ior);

        refractionRadiancePayload.colorWeight = radiancePayload.colorWeight * sbtData->dissolve;
        refractionRadiancePayload.depth = radiancePayload.depth + 1;
        traceRadiance(params.handle,
            geom.P + R * 0.01f,
            R,
            0.001f,
            1e16f,
            &refractionRadiancePayload
        );
        color += refractionRadiancePayload.color;
    }
    
   
    radiancePayload.color = color;
    
    setPayloadRadiance(radiancePayload);
}

extern "C" __global__ void __anyhit__occlusion()
{
    HitGroupData* sbtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    
    
    OcclusionPRD p = getPayloadOcclusion();
    p.isShadowed = false;
    setPayloadOcclusion(p);
    
}
