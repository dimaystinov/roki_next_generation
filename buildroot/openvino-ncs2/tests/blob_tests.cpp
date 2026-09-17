// SPDX-License-Identifier: Apache-2.0
#include "blob.hpp"
#include <iostream>
using namespace ncs2;
static void put(std::vector<uint8_t>& data, size_t pos, uint32_t value) {
    for (int i=0; i<4; ++i) data.at(pos+i) = (value >> (8*i)) & 255;
}
static std::vector<uint8_t> fixture() {
    std::vector<uint8_t> b(320);
    put(b, 52, 9709); put(b,56,b.size()); put(b,60,6); put(b,64,0);
    put(b,68,1); put(b,72,1); put(b,80,8); put(b,84,8);
    put(b,116,132); put(b,120,192); put(b,128,252);
    for (size_t start : {132,192}) {
        put(b,start,0); put(b,start+4,0); put(b,start+8,16);
        b[start+12]='x';
        put(b,start+28,1); put(b,start+32,0x21); put(b,start+36,2);
        put(b,start+40,3); put(b,start+44,0); put(b,start+48,3); put(b,start+52,8);
    }
    put(b,252,3); put(b,256,2); put(b,260,1); put(b,264,4);
    return b;
}
int main() {
    auto bytes = fixture(); Blob blob(bytes);
    require(blob.inputs[0].shape == std::vector<size_t>({2,3}), "physical shape order");
    std::vector<uint8_t> packed(8,99), host{1,2,3,4,5,6}, returned(6);
    transfer(blob.inputs[0],host.data(),packed.data(),true);
    require(packed == std::vector<uint8_t>({1,2,3,99,4,5,6,99}), "padding corrupted");
    transfer(blob.outputs[0],returned.data(),packed.data(),false);
    require(returned == host, "tensor roundtrip");
    int rejected=0;
    auto reject=[&](std::vector<uint8_t> b) {
        try { Blob invalid(std::move(b)); } catch(const std::runtime_error&) { ++rejected; return; }
        throw std::runtime_error("invalid blob accepted");
    };
    for (size_t size : {0,51,131,251,263}) { auto b=bytes; b.resize(size); reject(b); }
    for (const auto& edit : std::vector<std::pair<size_t,uint32_t>>{
          {52,1},{60,7},{64,1},{56,999},{68,0},{68,999},{80,0},{80,4},{116,0},
          {128,319},{140,5000},{160,9},{164,0x11},{168,9},{172,1},
          {176,0xffffffff},{184,0xffffffff},{252,0},{260,0},{264,1}}) {
        auto b=bytes; put(b,edit.first,edit.second); reject(b);
    }
    auto compact=bytes; put(compact,264,3); Blob cb(compact);
    require(cb.inputs[0].compact, "compact fast path not recognized");
    std::cout << "blob parser/transfer passed; malformed cases rejected: " << rejected << '\n';
}
