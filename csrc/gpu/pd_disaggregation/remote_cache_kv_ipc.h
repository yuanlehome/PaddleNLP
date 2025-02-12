#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "driver_types.h"
#include "paddle/extension.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"

struct RemoteCacheKvIpc {
    struct save_cache_kv_compelete_signal_layerwise_meta_data{
        int32_t layer_id=-1;
        void * shm_ptr=nullptr;
        int shm_fd=-1;
        save_cache_kv_compelete_signal_layerwise_meta_data(){}
        save_cache_kv_compelete_signal_layerwise_meta_data(int32_t layer_id_,
                                                            void* shm_ptr_,
                                                            int shm_fd_)
            :layer_id(layer_id_), shm_ptr(shm_ptr_), shm_fd(shm_fd_){
        }
    };
    static RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise_meta_data kv_compelete_signal_meta_data;
    static void* kv_compelete_signal_identity_ptr;
    static bool kv_compelete_signal_shmem_opened;

    static RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise_meta_data open_shm_and_get_compelete_signal_meta_data(const int rank_id);
    static void CUDART_CB save_cache_kv_compelete_signal_layerwise(void* meta_data);
};
