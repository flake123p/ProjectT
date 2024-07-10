#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdint.h>
// uintptr_t is defined in C++11 and later standards.
#define POINTER_TO_INT(ptr) ((uintptr_t)ptr)
#define POINTER_TO_U32(ptr) ((uint32_t)POINTER_TO_INT(ptr))

#define DUMPNC(a) printf(#a " = %c\n", a)
#define DUMPNS(a) printf(#a " = %s\n", a)
#define DUMPNSPP(a) printf(#a " = %s\n", a.c_str())
#define DUMPND(a) printf(#a " = %2d;\n", (int)(a))
#define DUMPNU(a) printf(#a " = %2u;\n", (unsigned int)a)
#define DUMPNX(a) printf(#a " = 0x%X\n", (unsigned int)(a))
#define DUMPNP(a) printf(#a " = %p\n", (void *)(a))
#define DUMPNA(a) printf(#a " = 0x%08X\n", POINTER_TO_U32(a))
#define DUMPNF(a) printf(#a " = %f\n", a)

#define DUMPNX8(a) printf(#a " = 0x%02X\n", (unsigned int)(a))
#define DUMPNX16(a) printf(#a " = 0x%04X\n", (unsigned int)(a))
#define DUMPNX32(a) printf(#a " = 0x%08X\n", (unsigned int)(a))
#define DUMPNX_LIBU64(a) DUMPNX32((a).hi);DUMPNX32((a).lo);

#define DUMPC(a) printf(#a " = %c, ", a)
#define DUMPS(a) printf(#a " = %s, ", a)
#define DUMPSPP(a) printf(#a " = %s, ", a.c_str())
#define DUMPD(a) printf(#a " = %2d, ", (int)(a))
#define DUMPU(a) printf(#a " = %2u, ", (unsigned int)a)
#define DUMPX(a) printf(#a " = 0x%X, ", (unsigned int)(a))
#define DUMPP(a) printf(#a " = %p, ", (void *)(a))
#define DUMPA(a) printf(#a " = 0x%08X, ", POINTER_TO_U32(a))
#define DUMPF(a) printf(#a " = %f, ", a)

#define DNC(a) DUMPNC(a)
#define DNS(a) DUMPNS(a)
#define DNSPP(a) DUMPNSPP(a)
#define DND(a) DUMPND(a)
#define DNU(a) DUMPNU(a)
#define DNX(a) DUMPNX(a)
#define DNP(a) DUMPNP(a)
#define DNA(a) DUMPNA(a)
#define DNF(a) DUMPNF(a)
#define DPC(a) DUMPC(a)
#define DPS(a) DUMPS(a)
#define DPSPP(a) DUMPSPP(a)
#define DPD(a) DUMPD(a)
#define DPU(a) DUMPU(a)
#define DPX(a) DUMPX(a)
#define DPP(a) DUMPP(a)
#define DPA(a) DUMPA(a)
#define DPF(a) DUMPF(a)

#define ARRAYDUMPC(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %c\n", xi, (a)[xi]);}
#define ARRAYDUMPS(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %s\n", xi, (a)[xi]);}
#define ARRAYDUMPD(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %2d;\n", xi, (a)[xi]);}
#define ARRAYDUMPU(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %2u\n", xi, (a)[xi]);}
#define ARRAYDUMPX(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = 0x%X;\n", xi, (uint8_t)(a)[xi]);}
#define ARRAYDUMPX2(a,length) printf(#a" = ");for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf("%02X ", (uint32_t)((a)[xi]));}printf("\n");
#define ARRAYDUMPX3(a,length) printf(#a" =\n");for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf("%02X ", (uint32_t)((a)[xi]));if(xi%16==15){printf("\n");}}printf("\n");
#define ARRAYDUMPP(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = %p\n", xi, (void *)(a)[xi]);}
#define ARRAYDUMPA(a,length) for(unsigned int xi=0; xi<(unsigned int)(length); xi++){printf(#a"[%d] = 0x%08X\n", xi, POINTER_TO_U32((a)[xi]));}

int main() 
{
    {
        cudaError_t rc;

        printf("cudaGetDeviceProperties\n");
        struct cudaDeviceProp pp;
        rc = cudaGetDeviceProperties(&pp, 0);
        printf("RC = %d\n", rc);
        printf("name = %s\n", pp.name);

        // for (int i = 0; i < 16; i++) {
        //     printf("%02X ", (unsigned char)pp.uuid.bytes[i]);
        // }
        // printf("\n");
        ARRAYDUMPX(pp.uuid.bytes, 16);

        DND(pp.luidDeviceNodeMask);
        DNU(pp.totalGlobalMem);
        DND(pp.sharedMemPerBlock);
        DND(pp.regsPerBlock);
        DND(pp.warpSize);
        DND(pp.memPitch);
        DND(pp.maxThreadsPerBlock);
        ARRAYDUMPD(pp.maxThreadsDim, 3);
        ARRAYDUMPD(pp.maxGridSize, 3);

        DND(pp.clockRate);
        DND(pp.totalConstMem);
        DND(pp.major);
        DND(pp.minor);
        DND(pp.textureAlignment);
        DND(pp.texturePitchAlignment);
        DND(pp.deviceOverlap);
        DND(pp.multiProcessorCount);
        DND(pp.kernelExecTimeoutEnabled);

        DND(pp.integrated);
        DND(pp.canMapHostMemory);
        DND(pp.computeMode);
        DND(pp.maxTexture1D);
        DND(pp.maxTexture1DMipmap);
        DND(pp.maxTexture1DLinear);

        ARRAYDUMPD(pp.maxTexture2D, 2);

        ARRAYDUMPD(pp.maxTexture2DMipmap, 2);
        ARRAYDUMPD(pp.maxTexture2DLinear, 3);
        ARRAYDUMPD(pp.maxTexture2DGather, 2);

        ARRAYDUMPD(pp.maxTexture3D, 3);
        ARRAYDUMPD(pp.maxTexture3DAlt, 3);
        DND(pp.maxTextureCubemap);
        ARRAYDUMPD(pp.maxTexture1DLayered, 2);
        ARRAYDUMPD(pp.maxTexture2DLayered, 3);
        ARRAYDUMPD(pp.maxTextureCubemapLayered, 2);
        DND(pp.maxSurface1D);
        ARRAYDUMPD(pp.maxSurface2D, 2);
        ARRAYDUMPD(pp.maxSurface3D, 3);

        ARRAYDUMPD(pp.maxSurface1DLayered, 2);
        ARRAYDUMPD(pp.maxSurface2DLayered, 3);

        DND(pp.maxSurfaceCubemap);
        ARRAYDUMPD(pp.maxSurfaceCubemapLayered, 2);

        DND(pp.surfaceAlignment);

        DND(pp.concurrentKernels);
        DND(pp.ECCEnabled);
        DND(pp.pciBusID);
        DND(pp.pciDeviceID);
        DND(pp.pciDomainID);
        DND(pp.tccDriver);
        DND(pp.asyncEngineCount);
        DND(pp.unifiedAddressing);
        DND(pp.memoryClockRate);
        DND(pp.memoryBusWidth);
        DND(pp.l2CacheSize);
        DND(pp.persistingL2CacheMaxSize);
        DND(pp.maxThreadsPerMultiProcessor);
        DND(pp.streamPrioritiesSupported);
        DND(pp.globalL1CacheSupported);
        DND(pp.localL1CacheSupported);
        DND(pp.sharedMemPerMultiprocessor);
        DND(pp.regsPerMultiprocessor);
        DND(pp.managedMemory);
        DND(pp.isMultiGpuBoard);
        DND(pp.multiGpuBoardGroupID);
        DND(pp.hostNativeAtomicSupported);
        DND(pp.singleToDoublePrecisionPerfRatio);
        DND(pp.pageableMemoryAccess);
        DND(pp.concurrentManagedAccess);

        DND(pp.computePreemptionSupported);
        DND(pp.canUseHostPointerForRegisteredMem);
        DND(pp.cooperativeLaunch);
        DND(pp.cooperativeMultiDeviceLaunch);
        DND(pp.sharedMemPerBlockOptin);
        DND(pp.pageableMemoryAccessUsesHostPageTables);
        DND(pp.directManagedMemAccessFromHost);
        DND(pp.maxBlocksPerMultiProcessor);
        DND(pp.accessPolicyMaxWindowSize);
        DND(pp.reservedSharedMemPerBlock);  
    }

    return 0;
}