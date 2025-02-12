#include "remote_cache_kv_ipc.h"

RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise_meta_data RemoteCacheKvIpc::kv_compelete_signal_meta_data;
void* RemoteCacheKvIpc::kv_compelete_signal_identity_ptr = nullptr;
bool RemoteCacheKvIpc::kv_compelete_signal_shmem_opened = false;

RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise_meta_data RemoteCacheKvIpc::open_shm_and_get_compelete_signal_meta_data(const int rank_id){
    if(RemoteCacheKvIpc::kv_compelete_signal_shmem_opened){
        int32_t current_identity = (*reinterpret_cast<int32_t*>(RemoteCacheKvIpc::kv_compelete_signal_identity_ptr));
        int32_t* write_ptr = reinterpret_cast<int32_t*>(RemoteCacheKvIpc::kv_compelete_signal_identity_ptr);
        *write_ptr = (current_identity + 1) % 100003;
        RemoteCacheKvIpc::kv_compelete_signal_meta_data.layer_id = -1;        
        int32_t* layer_complete_ptr = reinterpret_cast<int32_t*>(kv_compelete_signal_meta_data.shm_ptr);
        *layer_complete_ptr = -1;
        return RemoteCacheKvIpc::kv_compelete_signal_meta_data;
    }
    std::string flags_server_uuid;
    if (const char* iflags_server_uuid_env_p = std::getenv("SHM_UUID")){
        std::string iflags_server_uuid_env_str(iflags_server_uuid_env_p);
        flags_server_uuid = iflags_server_uuid_env_str;
    }
    std::string signal_shm_name_string = "splitwise_complete_prefilled_layer_" 
                                            + std::to_string(rank_id) + "_" + flags_server_uuid;
    int signal_shm_fd = shm_open(signal_shm_name_string.c_str(), O_CREAT | O_RDWR, 0666);

    PADDLE_ENFORCE_NE(signal_shm_fd,
                        -1,
                        phi::errors::InvalidArgument(
                            "can not open shm for cache_kv_compelete_signal."));
    int signal_shm_ftruncate = ftruncate(signal_shm_fd, 4); 
    void* signal_ptr = mmap(0, 4, PROT_WRITE, MAP_SHARED, signal_shm_fd, 0);

    PADDLE_ENFORCE_NE(
        signal_ptr,
        MAP_FAILED,
        phi::errors::InvalidArgument(
                            "MAP_FAILED for cache_kv_compelete_signal."));
    int32_t* write_signal_ptr = reinterpret_cast<int32_t*>(signal_ptr);
    *write_signal_ptr = -1;
    using type_meta_data = RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise_meta_data;

    // std::printf("#### open_shm_and_get_compelete_signal_meta_data layer idx:%d, to ptx:%p \n", 
    //             -1, signal_ptr);

    type_meta_data meta_data(
        -1,
        signal_ptr,
        signal_shm_fd
    );
    RemoteCacheKvIpc::kv_compelete_signal_meta_data = meta_data;
    std::string prefill_identity_shm_name_string = "splitwise_complete_prefilled_step_" 
                                    + std::to_string(rank_id) + "_" + flags_server_uuid;
    int identity_shm_fd = shm_open(prefill_identity_shm_name_string.c_str(), O_CREAT | O_RDWR, 0666);
    PADDLE_ENFORCE_NE(identity_shm_fd,
                        -1,
                        phi::errors::InvalidArgument(
                            "can not open shm for cache_kv_compelete_identity."));

    int identity_shm_ftruncate = ftruncate(identity_shm_fd, 4); 
    void* identity_ptr = mmap(0, 4, PROT_WRITE, MAP_SHARED, identity_shm_fd, 0);
    PADDLE_ENFORCE_NE(
        identity_ptr,
        MAP_FAILED,
        phi::errors::InvalidArgument(
                            "MAP_FAILED for prefill_identity."));
    int32_t current_identity = (*reinterpret_cast<int32_t*>(identity_ptr));
    int32_t* write_ptr = reinterpret_cast<int32_t*>(identity_ptr);
    *write_ptr = (current_identity + 1) % 100003;
    RemoteCacheKvIpc::kv_compelete_signal_identity_ptr = identity_ptr;
    RemoteCacheKvIpc::kv_compelete_signal_shmem_opened = true;
    return meta_data;
}

void CUDART_CB RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise(void* meta_data){
    int64_t* meta_data_ptr = reinterpret_cast<int64_t*>(meta_data);
    int32_t layer_id = meta_data_ptr[0];
    int32_t* ptr = reinterpret_cast<int32_t*>(meta_data_ptr[1]);
    *ptr = layer_id;
    // std::printf("#### save_cache_kv_compelete_signal_layerwise layer idx:%d, to ptx:%p \n", 
    //             *ptr, meta_data_ptr[1]);
}