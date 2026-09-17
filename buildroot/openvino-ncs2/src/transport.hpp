// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "blob.hpp"
#include "mvnc.h"
#include <mutex>

namespace ncs2 {
inline void check(ncStatus_t code, const char* operation) {
    require(code == NC_OK, std::string(operation) + " failed: mvnc status " + std::to_string(code));
}
inline std::mutex& transport_mutex() { static std::mutex mutex; return mutex; }
inline std::vector<std::string> devices() {
    std::lock_guard<std::mutex> lock(transport_mutex());
    ncDeviceDescr_t descriptors[NC_MAX_DEVICES]{};
    int count = 0;
    check(ncAvailableDevices(descriptors, NC_MAX_DEVICES, &count), "enumerate USB devices");
    require(count >= 0 && count <= NC_MAX_DEVICES, "invalid device count");
    std::vector<std::string> names;
    for (int i=0; i<count; ++i)
        if (descriptors[i].protocol == NC_USB)
            names.emplace_back(descriptors[i].name, strnlen(descriptors[i].name, NC_MAX_NAME_SIZE));
    return names;
}
class Session {
    WatchdogHndl_t* watchdog = nullptr;
    ncDeviceHandle_t* device = nullptr;
    ncGraphHandle_t* graph = nullptr;
    ncFifoHandle_t* input = nullptr;
    ncFifoHandle_t* output = nullptr;
    bool failed = false;
    void close() noexcept {
        if (input) ncFifoDestroy(&input);
        if (output) ncFifoDestroy(&output);
        if (graph) ncGraphDestroy(&graph);
        if (device) ncDeviceClose(&device, watchdog);
        if (watchdog) watchdog_destroy(watchdog);
        watchdog = nullptr;
    }
public:
    Session(const Blob& blob, const std::string& firmware, const std::string& id) {
        std::lock_guard<std::mutex> lock(transport_mutex());
        try {
            require(watchdog_create(&watchdog) == WD_ERRNO, "watchdog initialization failed");
            int timeout = 10000;
            check(ncGlobalSetOption(NC_RW_COMMON_TIMEOUT_MSEC, &timeout, sizeof(timeout)), "set I/O timeout");
            check(ncSetDeviceConnectTimeout(10), "set device timeout");
            ncDeviceDescr_t description{}; description.protocol = NC_USB;
            require(id.size() < sizeof(description.name), "device id too long");
            std::memcpy(description.name, id.data(), id.size());
            ncDeviceOpenParams_t params{};
            params.watchdogHndl = watchdog; params.watchdogInterval = 1000;
            params.customFirmwareDirectory = firmware.c_str();
            check(ncDeviceOpen(&device, description, params), "open NCS2 (check USB access and firmware)");
            check(ncGraphCreate("roki-ncs2", &graph), "create graph");
            int executors = 1;
            check(ncGraphSetOption(graph, NC_RW_GRAPH_EXECUTORS_NUM, &executors, sizeof(executors)), "set executors");
            check(ncGraphAllocate(device, graph, blob.data.data(), blob.file_size,
                                 blob.data.data(), Blob::header_size), "upload graph");
            int count = 0; unsigned int length = sizeof(count);
            check(ncGraphGetOption(graph, NC_RO_GRAPH_INPUT_COUNT, &count, &length), "input FIFO count");
            require(count == 1, "expected one packed input FIFO");
            length = sizeof(count);
            check(ncGraphGetOption(graph, NC_RO_GRAPH_OUTPUT_COUNT, &count, &length), "output FIFO count");
            require(count == 1, "expected one packed output FIFO");
            ncTensorDescriptor_t in{}, out{};
            length = sizeof(in);
            check(ncGraphGetOption(graph, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &in, &length), "input descriptor");
            length = sizeof(out);
            check(ncGraphGetOption(graph, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &out, &length), "output descriptor");
            require(in.totalSize == blob.input_size && out.totalSize == blob.output_size, "firmware I/O size mismatch");
            check(ncFifoCreate("input", NC_FIFO_HOST_WO, &input), "create input FIFO");
            check(ncFifoAllocate(input, device, &in, 2), "allocate input FIFO");
            check(ncFifoCreate("output", NC_FIFO_HOST_RO, &output), "create output FIFO");
            check(ncFifoAllocate(output, device, &out, 2), "allocate output FIFO");
        } catch (...) { close(); throw; }
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session() { std::lock_guard<std::mutex> lock(transport_mutex()); close(); }
    void infer(const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
        // Queue and read must be one transaction across all async requests.
        std::lock_guard<std::mutex> lock(transport_mutex());
        require(!failed, "request stream failed; reload model to recover");
        try {
            auto size = static_cast<unsigned int>(in.size());
            check(ncGraphQueueInferenceWithFifoElem(graph, input, output, in.data(), &size, nullptr), "queue inference");
            require(size == in.size(), "partial input write");
            size = static_cast<unsigned int>(out.size()); void* token = nullptr;
            check(ncFifoReadElem(output, out.data(), &size, &token), "read inference result");
            require(size == out.size(), "partial output read");
        } catch (...) { failed = true; throw; }
    }
};
}  // namespace ncs2
