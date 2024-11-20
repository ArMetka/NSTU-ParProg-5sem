#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>

#include "./solution.h"

int main(int argc, char **argv) {
    if (get_devices_info()) return 1;

    return 0;
}

int get_devices_info() {
    cl_platform_id platforms[64];
    unsigned int platformCount;
    cl_int platformsResult = clGetPlatformIDs(64, platforms, &platformCount);
    
    if (platformsResult) return 1;

    for (int i = 0; i < platformCount; i++) {
        cl_device_id devices[64];
        unsigned int deviceCount;
        cl_int devicesResult = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);
        if (devicesResult) return 1;

        for (int j = 0; j < deviceCount; j++) {
            char vendorName[256];
            char deviceName[256];
            char deviceDriver[256];
            size_t vendorNameLen;
            size_t deviceNameLen;
            size_t deviceDriverLen;

            cl_int infoResult = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 256, vendorName, &vendorNameLen);
            if (infoResult) return 1;
            infoResult = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 256, deviceName, &deviceNameLen);
            if (infoResult) return 1;
            infoResult = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 256, deviceDriver, &deviceDriverLen);
            if (infoResult) return 1;

            printf("Vendor Name: %s\nDevice Name: %s\nDevice Driver: %s\n\n", vendorName, deviceName, deviceDriver);

        }
    }

    return 0;
}