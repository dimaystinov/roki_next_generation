// SPDX-License-Identifier: Apache-2.0
// Wire format: OpenVINO 2022.3.2 vpu/backend/blob_format.hpp and blob_reader.cpp.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace ncs2 {
inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("NCS2: " + message);
}
struct Port {
    std::string name;
    uint32_t type = 0, offset = 0;
    size_t item_size = 0, bytes = 0, span = 0;
    // Physical axis order, outermost first. No implicit NCHW/NHWC conversion.
    std::vector<size_t> shape, strides;
    bool compact = false;
};
struct Blob {
    static constexpr size_t header_size = 52 + 80;
    static constexpr size_t max_size = 256 * 1024 * 1024;
    std::vector<uint8_t> data;
    uint32_t file_size = 0, input_size = 0, output_size = 0;
    std::vector<Port> inputs, outputs;

    uint32_t word(size_t pos) const {
        require(pos <= file_size && file_size - pos >= 4, "truncated blob field");
        return uint32_t(data[pos]) | (uint32_t(data[pos+1]) << 8) |
               (uint32_t(data[pos+2]) << 16) | (uint32_t(data[pos+3]) << 24);
    }
    explicit Blob(std::vector<uint8_t> bytes) : data(std::move(bytes)) {
        require(data.size() >= header_size && data.size() <= max_size, "invalid blob length");
        file_size = static_cast<uint32_t>(data.size());
        require(word(52) == 9709 && word(60) == 6 && word(64) == 0, "expected MYRIAD blob format 6.0");
        auto declared = word(56);
        require(declared >= header_size && declared <= data.size(), "invalid declared file size");
        file_size = declared;
        input_size = word(80); output_size = word(84);
        require(input_size && output_size && input_size <= max_size && output_size <= max_size,
                "invalid packed I/O buffer size");
        const auto constant_section = word(128);
        require(constant_section >= header_size && constant_section <= file_size, "invalid constant section");
        inputs = parse_ports(word(68), word(116), input_size, constant_section);
        outputs = parse_ports(word(72), word(120), output_size, constant_section);
    }
    std::vector<Port> parse_ports(uint32_t count, size_t pos, size_t buffer_size, size_t constants) const {
        require(count > 0 && count <= 128, "unsupported I/O count");
        require(pos >= header_size && pos < constants, "invalid I/O section");
        std::vector<Port> ports;
        std::set<std::string> names;
        for (uint32_t i = 0; i < count; ++i) {
            auto next = [&]() { auto value = word(pos); pos += 4; return value; };
            require(next() == i, "invalid I/O index");
            Port p;
            p.offset = next();
            const size_t length = next();
            require(length > 0 && length <= 4096 && pos <= constants && length <= constants-pos,
                    "invalid tensor name");
            auto begin = reinterpret_cast<const char*>(data.data()+pos);
            auto end = static_cast<const char*>(std::memchr(begin, 0, length));
            require(end != nullptr && end != begin, "unterminated or empty tensor name");
            p.name.assign(begin, end); pos += length;
            require(names.insert(p.name).second, "duplicate tensor name");
            require(p.name.find("@shape") == std::string::npos, "dynamic shape tensors unsupported");
            p.type = next();
            const size_t sizes[] = {2, 1, 4, 4, 1};
            require(p.type < 5, "unsupported element type"); p.item_size = sizes[p.type];
            uint32_t order = next(), rank = next();
            require(rank > 0 && rank <= 8, "invalid tensor rank");
            require(next() == 3, "shape must reside in blob");
            size_t dimensions = next();
            require(dimensions <= file_size-constants, "invalid dimensions offset");
            dimensions += constants;
            require(next() == 3, "strides must reside in blob");
            size_t strides = next();
            require(strides <= file_size-constants, "invalid strides offset");
            strides += constants;
            require(pos <= constants, "I/O section overlaps constants");
            size_t elements = 1, span = p.item_size, inner_span = p.item_size;
            unsigned seen = 0;
            for (uint32_t axis = 0; axis < rank; ++axis) {
                unsigned id = order & 15; order >>= 4;
                require(id >= 1 && id <= 8 && !(seen & (1u << id)), "invalid axis order");
                seen |= 1u << id;
                size_t dim = word(dimensions + axis*4), stride = word(strides + axis*4);
                require(dim > 0 && elements <= max_size / p.item_size / dim, "invalid tensor dimensions");
                require(stride >= inner_span && stride <= max_size && dim <= max_size/stride,
                        "invalid or overlapping tensor strides");
                elements *= dim; span += (dim-1)*stride; inner_span = dim*stride;
                p.shape.insert(p.shape.begin(), dim); p.strides.insert(p.strides.begin(), stride);
            }
            require(order == 0, "rank and order disagree");
            p.bytes = elements * p.item_size; p.span = span; p.compact = span == p.bytes;
            require(p.offset <= buffer_size && span <= buffer_size-p.offset, "tensor outside I/O buffer");
            for (const auto& old : ports)
                require(size_t(p.offset)+span <= old.offset || size_t(old.offset)+old.span <= p.offset,
                        "overlapping I/O tensors unsupported");
            ports.push_back(std::move(p));
        }
        return ports;
    }
};

// Copy between compact host tensors and device layout, retaining padding.
inline void transfer(const Port& p, void* host, uint8_t* packed, bool to_device) {
    auto* h = static_cast<uint8_t*>(host);
    packed += p.offset;
    if (p.compact) {
        if (to_device) std::memcpy(packed, h, p.bytes);
        else std::memcpy(h, packed, p.bytes);
        return;
    }
    for (size_t index = 0; index < p.bytes/p.item_size; ++index) {
        size_t remainder = index, offset = 0;
        for (size_t axis = p.shape.size(); axis-- > 0;) {
            offset += (remainder % p.shape[axis]) * p.strides[axis];
            remainder /= p.shape[axis];
        }
        if (to_device) std::memcpy(packed+offset, h+index*p.item_size, p.item_size);
        else std::memcpy(h+index*p.item_size, packed+offset, p.item_size);
    }
}
}  // namespace ncs2
