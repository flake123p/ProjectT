//
// Template argument deduction
//      https://en.cppreference.com/w/cpp/language/template_argument_deduction
//
// https://stackoverflow.com/questions/10872730/can-a-template-function-be-called-with-missing-template-parameters-in-c
//


#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <numeric>
#include <cmath>
#include <map>
#include <thread>
#include <algorithm>
#include <future>
#include <climits>
#include <cfloat>
#include <typeinfo>
#include <cstring>

#define COUT(a) std::cout << #a " = " << a << std::endl
#define PRINT_FUNC printf("%s()\n", __func__);

typedef enum {
    //vendor
    CMD_REQ_TYPE_VENDOR = 0x0000,
    CMD_REQ_ECHO,
    
    //system (RT API)
    CMD_REQ_TYPE_CU_RT = 0x1000,
    CMD_REQ_MALLOC,
    CMD_REQ_FREE,
} CMD_REQ_t;

typedef struct {
    uint16_t cmd_id; //CMD_REQ_t
    uint16_t cmd_handle;
    uint8_t data_type; //CMD_DATA_TYPE_t
    uint8_t RESERVED_0;
    uint16_t RESERVED_1;
} CMD_REQ_HDR_t; //Request Header

typedef struct {
    uint16_t cmd_id; //CMD_REQ_t
    uint16_t cmd_handle;
    int32_t status;
} CMD_RES_HDR_t; //Response Header

typedef struct {
    CMD_REQ_HDR_t req_hdr;
    uint64_t size;
} CMD_REQ_MALLOC_t; //__host____device__cudaError_t cudaMalloc (void **devPtr, size_t size)

typedef struct {
    CMD_RES_HDR_t res_hdr;
    uint64_t addr;
} CMD_RES_MALLOC_t; //response

typedef enum {
    CMD_DTYPE_NONE = 0,
    CMD_DTYPE_FP32,
    CMD_DTYPE_FP64,
    CMD_DTYPE_FP16,
    CMD_DTYPE_BP16,
    CMD_DTYPE_INT64,
    CMD_DTYPE_INT32,
    CMD_DTYPE_INT16,
    CMD_DTYPE_INT8,
    CMD_DTYPE_UINT64,
    CMD_DTYPE_UINT32,
    CMD_DTYPE_UINT16,
} CMD_DATA_TYPE_t;

template<uint32_t id, typename Q_TYPE, typename S_TYPE, uint8_t dtype = CMD_DTYPE_NONE>
class ReqRes {
public:
    uint32_t x = id;
    Q_TYPE req;
    S_TYPE res;
    ReqRes() {
        CMD_REQ_HDR_t *req_hdr = reinterpret_cast<CMD_REQ_HDR_t *>(&req);
        CMD_RES_HDR_t *res_hdr = reinterpret_cast<CMD_RES_HDR_t *>(&res);
        
        memset(&req, 0, sizeof(Q_TYPE));
        req_hdr->cmd_id = id;
        req_hdr->data_type = dtype;
    }

};

#define EXPAND(name)  CMD_REQ_##name, CMD_REQ_##name##_t, CMD_RES_##name##_t
#define EXPAND2(name) CMD_REQ_##name, CMD_REQ_##name##_t, CMD_RES_HDR_t

int main()
{
    // use case 1:
    //auto qs = ReqRes<CMD_REQ_MALLOC, CMD_REQ_MALLOC_t, CMD_RES_MALLOC_t>();

    // use case 2:
    //auto qs = ReqRes<EXPAND(MALLOC)>();

    // use case 3:
    auto qs = ReqRes<EXPAND2(MALLOC)>();

    COUT(qs.x);

    return 0;
}