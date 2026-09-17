// SPDX-License-Identifier: Apache-2.0
// Test-only transport. Never linked into or installed with the production plugin.
#include "mvnc.h"
#include <cstring>
#include <vector>
struct _WatchdogHndl_t {};
struct _graphPrivate_t { unsigned in=0, out=0; };
struct _fifoPrivate_t { std::vector<unsigned char> bytes; };
extern "C" {
wd_error_t watchdog_create(WatchdogHndl_t** p) { *p=new _WatchdogHndl_t; return WD_ERRNO; }
void watchdog_destroy(WatchdogHndl_t* p) { delete p; }
ncStatus_t ncGlobalSetOption(ncGlobalOption_t,const void*,unsigned int) { return NC_OK; }
ncStatus_t ncSetDeviceConnectTimeout(int) { return NC_OK; }
ncStatus_t ncAvailableDevices(ncDeviceDescr_t* d,int,int* count) {
    *count=1; d[0].protocol=NC_USB; std::strcpy(d[0].name,"TEST-NOT-A-USB-DEVICE"); return NC_OK;
}
ncStatus_t ncDeviceOpen(ncDeviceHandle_t** d,ncDeviceDescr_t,ncDeviceOpenParams_t) {
    *d=new ncDeviceHandle_t{}; return NC_OK;
}
ncStatus_t ncDeviceClose(ncDeviceHandle_t** d,WatchdogHndl_t*) { delete *d; *d=nullptr; return NC_OK; }
ncStatus_t ncGraphCreate(const char*,ncGraphHandle_t** g) { *g=new ncGraphHandle_t{new _graphPrivate_t}; return NC_OK; }
ncStatus_t ncGraphDestroy(ncGraphHandle_t** g) { delete (*g)->private_data; delete *g; *g=nullptr; return NC_OK; }
ncStatus_t ncGraphSetOption(ncGraphHandle_t*,ncGraphOption_t,const void*,unsigned int) { return NC_OK; }
ncStatus_t ncGraphAllocate(ncDeviceHandle_t*,ncGraphHandle_t* g,const void*,unsigned int,
                         const void* header,unsigned int length) {
    if (length!=132) return NC_INVALID_DATA_LENGTH;
    auto* h=static_cast<const unsigned char*>(header);
    auto read=[&](int pos) { return unsigned(h[pos])|(unsigned(h[pos+1])<<8)|(unsigned(h[pos+2])<<16)|(unsigned(h[pos+3])<<24); };
    g->private_data->in=read(80); g->private_data->out=read(84); return NC_OK;
}
ncStatus_t ncGraphGetOption(ncGraphHandle_t* g,ncGraphOption_t opt,void* p,unsigned int*) {
    if (opt==NC_RO_GRAPH_INPUT_COUNT || opt==NC_RO_GRAPH_OUTPUT_COUNT) *static_cast<int*>(p)=1;
    else {
        auto* desc=static_cast<ncTensorDescriptor_t*>(p); *desc={};
        desc->totalSize=opt==NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS ? g->private_data->in : g->private_data->out;
    }
    return NC_OK;
}
ncStatus_t ncFifoCreate(const char*,ncFifoType_t,ncFifoHandle_t** f) { *f=new ncFifoHandle_t{new _fifoPrivate_t}; return NC_OK; }
ncStatus_t ncFifoAllocate(ncFifoHandle_t* f,ncDeviceHandle_t*,ncTensorDescriptor_t* d,unsigned int) {
    f->private_data->bytes.resize(d->totalSize); return NC_OK;
}
ncStatus_t ncFifoDestroy(ncFifoHandle_t** f) { delete (*f)->private_data; delete *f; *f=nullptr; return NC_OK; }
ncStatus_t ncGraphQueueInferenceWithFifoElem(ncGraphHandle_t*,ncFifoHandle_t*,ncFifoHandle_t* out,
                                          const void* input,unsigned int* size,void*) {
    auto* bytes=static_cast<const unsigned char*>(input);
    if (bytes[0]==255) return NC_TIMEOUT;
    if (*size!=out->private_data->bytes.size()) return NC_INVALID_DATA_LENGTH;
    std::memcpy(out->private_data->bytes.data(),input,*size); return NC_OK;
}
ncStatus_t ncFifoReadElem(ncFifoHandle_t* f,void* output,unsigned int* size,void**) {
    if (*size!=f->private_data->bytes.size()) return NC_INVALID_DATA_LENGTH;
    std::memcpy(output,f->private_data->bytes.data(),*size); return NC_OK;
}
}
