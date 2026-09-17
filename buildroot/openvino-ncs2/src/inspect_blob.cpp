// SPDX-License-Identifier: Apache-2.0
#include "blob.hpp"
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: inspect_blob model.blob\n"; return 2; }
    try {
        std::ifstream in(argv[1],std::ios::binary|std::ios::ate);
        ncs2::require(bool(in),"cannot open blob");
        auto length=in.tellg();
        ncs2::require(length>=0 && size_t(length)<=ncs2::Blob::max_size,"invalid blob length");
        std::vector<uint8_t> data(static_cast<size_t>(length));
        in.seekg(0); in.read(reinterpret_cast<char*>(data.data()),data.size());
        ncs2::require(bool(in),"failed to read blob");
        ncs2::Blob blob(std::move(data));
        const char* types[]={"FP16","U8","I32","FP32","I8"};
        auto print=[&](const char* direction,const std::vector<ncs2::Port>& ports) {
            for (const auto& p:ports) {
                std::cout << direction << ' ' << p.name << ' ' << types[p.type] << " shape=";
                for (auto dim:p.shape) std::cout << dim << ',';
                std::cout << " offset=" << p.offset << " bytes=" << p.bytes
                          << " compact=" << p.compact << '\n';
            }
        };
        print("input",blob.inputs); print("output",blob.outputs);
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
