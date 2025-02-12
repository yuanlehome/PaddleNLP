#include "helper.h"
#include "cuda_multiprocess.h"


namespace {
    int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
    info->size = sz;
    info->shmFd = shm_open(name, O_RDWR, 0777);
    if (info->shmFd < 0) {
        return errno;
    }

    info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
    if (info->addr == NULL) {
        return errno;
    }

    return 0;
    }
}

std::vector<paddle::Tensor>  GetDataPtrIpc(const paddle::Tensor& tmp_input,
                   const std::string& shm_name) {
    auto out_data_ptr_tensor = paddle::full({1}, 0,  paddle::DataType::INT64, paddle::CPUPlace());
    auto out_data_ptr_tensor_ptr = out_data_ptr_tensor.data<int64_t>();
    volatile shmStruct *shm = NULL;
    sharedMemoryInfo info;
    if (sharedMemoryOpen(shm_name.c_str(), sizeof(shmStruct), &info) != 0) {
        printf("Failed to create shared memory slab\n");
        exit(EXIT_FAILURE);
    }
    shm = (volatile shmStruct *)info.addr;
    void *ptr = nullptr;
    checkCudaErrors(
        cudaIpcOpenMemHandle(&ptr, *(cudaIpcMemHandle_t *)&shm->memHandle,
                                cudaIpcMemLazyEnablePeerAccess));

    out_data_ptr_tensor_ptr[0]=reinterpret_cast<int64_t>(ptr);
    return {out_data_ptr_tensor};
}

PD_BUILD_OP(get_data_ptr_ipc)
    .Inputs({"tmp_input"})
    .Attrs({ "shm_name: std::string"})
    .Outputs({"data_ptr"})
    .SetKernelFn(PD_KERNEL(GetDataPtrIpc));