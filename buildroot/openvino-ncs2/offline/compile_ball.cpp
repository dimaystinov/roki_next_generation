// SPDX-License-Identifier: Apache-2.0
// Host-only model preparation. Never linked into the modern runtime/plugin.
#include <fstream>
#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "Usage: compile_ball model.xml output.blob\n"; return 2; }
    try {
        ov::Core core;
        auto model = core.read_model(argv[1]);
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().preprocess().convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB).scale({255.f,255.f,255.f});
        ppp.input().model().set_layout("NCHW");
        ppp.output().tensor().set_element_type(ov::element::f32);
        model = ppp.build();
        auto compiled = core.compile_model(model,"MYRIAD", {{"MYRIAD_ENABLE_MX_BOOT", "NO"}});
        std::ofstream out(argv[2],std::ios::binary);
        if (!out) throw std::runtime_error("Cannot create output file");
        compiled.export_model(out);
        if (!out) throw std::runtime_error("Blob export failed");
        std::cout << "Exported BGR U8 blob; no NCS2 device used\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
