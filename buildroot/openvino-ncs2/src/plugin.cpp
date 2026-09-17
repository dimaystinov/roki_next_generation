// SPDX-License-Identifier: Apache-2.0
#include "blob.hpp"
#include "transport.hpp"
#include <cstdlib>
#include <istream>
#include <ostream>
#include "openvino/op/op.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/properties.hpp"

namespace ncs2 {
static ov::element::Type element(const Port& port) {
    static const ov::element::Type types[] = {ov::element::f16, ov::element::u8, ov::element::i32,
                                              ov::element::f32, ov::element::i8};
    return types[port.type];
}
// Metadata representation of the opaque compiled device graph, not a CPU op.
class BlobOp : public ov::op::Op {
    std::vector<Port> ports;
public:
    OPENVINO_OP("NCS2Blob", "roki");
    BlobOp(const ov::OutputVector& args, std::vector<Port> outputs) : Op(args), ports(std::move(outputs)) {
        constructor_validate_and_infer_types();
    }
    void validate_and_infer_types() override {
        set_output_size(ports.size());
        for (size_t i=0; i<ports.size(); ++i) set_output_type(i, element(ports[i]), ov::Shape(ports[i].shape));
    }
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& args) const override {
        return std::make_shared<BlobOp>(args, ports);
    }
};
static std::shared_ptr<ov::Model> metadata_model(const Blob& blob) {
    ov::ParameterVector params; ov::OutputVector args; ov::ResultVector results;
    for (const auto& port : blob.inputs) {
        auto node = std::make_shared<ov::op::v0::Parameter>(element(port), ov::Shape(port.shape));
        node->set_friendly_name(port.name); node->output(0).set_names({port.name});
        params.push_back(node); args.push_back(node);
    }
    auto opaque = std::make_shared<BlobOp>(args, blob.outputs);
    for (size_t i=0; i<blob.outputs.size(); ++i) {
        opaque->output(i).set_names({blob.outputs[i].name});
        results.push_back(std::make_shared<ov::op::v0::Result>(opaque->output(i)));
    }
    return std::make_shared<ov::Model>(results, params, "NCS2 imported blob");
}
struct Config {
    std::string firmware, id;
    Config() { if (auto* path = std::getenv("NCS2_FIRMWARE_DIR")) firmware = path; }
    void update(const ov::AnyMap& values) {
        for (const auto& pair : values) {
            if (pair.first == "NCS2_FIRMWARE_DIR") firmware = pair.second.as<std::string>();
            else if (pair.first == ov::device::id.name()) id = pair.second.as<std::string>();
            else OPENVINO_THROW("NCS2: unsupported property ", pair.first);
        }
    }
};
class Request;
class Compiled : public ov::ICompiledModel {
    std::shared_ptr<ov::Model> model;
public:
    std::shared_ptr<const Blob> blob;
    mutable Session session;
    Compiled(std::shared_ptr<const Blob> b, const std::shared_ptr<const ov::IPlugin>& plugin, const Config& config)
        : ICompiledModel(metadata_model(*b), plugin), model(metadata_model(*b)), blob(b),
          session(*b, config.firmware, config.id) {}
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    void export_model(std::ostream& out) const override {
        out.write(reinterpret_cast<const char*>(blob->data.data()), blob->data.size());
        require(bool(out), "could not export blob");
    }
    std::shared_ptr<const ov::Model> get_runtime_model() const override { return model->clone(); }
    void set_property(const ov::AnyMap& values) override {
        require(values.empty(), "compiled model properties are read-only");
    }
    ov::Any get_property(const std::string& name) const override {
        if (name == ov::supported_properties.name())
            return std::vector<ov::PropertyName>{ov::supported_properties, ov::model_name,
                   ov::optimal_number_of_infer_requests, ov::execution_devices, ov::loaded_from_cache};
        if (name == ov::model_name.name()) return model->get_friendly_name();
        if (name == ov::optimal_number_of_infer_requests.name()) return unsigned(1);
        if (name == ov::execution_devices.name()) return std::vector<std::string>{"MYRIAD"};
        if (name == ov::loaded_from_cache.name()) return false;
        OPENVINO_THROW("NCS2: unsupported compiled-model property ", name);
    }
};
class Request : public ov::ISyncInferRequest {
    std::shared_ptr<const Compiled> compiled;
    std::vector<uint8_t> input, output;
public:
    explicit Request(const std::shared_ptr<const Compiled>& model)
        : ISyncInferRequest(model), compiled(model), input(model->blob->input_size), output(model->blob->output_size) {
        for (const auto& port : get_inputs()) allocate_tensor(port, [port](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(port.get_element_type(), port.get_shape());
        });
        for (const auto& port : get_outputs()) allocate_tensor(port, [port](ov::SoPtr<ov::ITensor>& tensor) {
            tensor = ov::make_tensor(port.get_element_type(), port.get_shape());
        });
    }
    void infer() override {
        check_tensors();
        std::fill(input.begin(), input.end(), 0);
        for (size_t i=0; i<get_inputs().size(); ++i) {
            auto tensor = get_tensor(get_inputs()[i]);
            require(tensor->is_continuous(), "non-contiguous input tensor unsupported");
            transfer(compiled->blob->inputs[i], tensor->data(), input.data(), true);
        }
        for (const auto& port : get_outputs())
            require(get_tensor(port)->is_continuous(), "non-contiguous output tensor unsupported");
        compiled->session.infer(input, output);
        for (size_t i=0; i<get_outputs().size(); ++i) {
            auto tensor = get_tensor(get_outputs()[i]);
            transfer(compiled->blob->outputs[i], tensor->data(), output.data(), false);
        }
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override { return {}; }
    std::vector<ov::ProfilingInfo> get_profiling_info() const override { return {}; }
};
std::shared_ptr<ov::ISyncInferRequest> Compiled::create_sync_infer_request() const {
    return std::make_shared<Request>(std::static_pointer_cast<const Compiled>(shared_from_this()));
}
class Plugin : public ov::IPlugin {
    mutable std::mutex config_mutex;
    Config config;
    std::shared_ptr<ov::ICompiledModel> load(std::vector<uint8_t> data, const ov::AnyMap& properties) const {
        Config cfg;
        { std::lock_guard<std::mutex> lock(config_mutex); cfg = config; }
        cfg.update(properties);
        return std::make_shared<Compiled>(std::make_shared<Blob>(std::move(data)), shared_from_this(), cfg);
    }
public:
    Plugin() { set_device_name("MYRIAD"); }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                                                     const ov::AnyMap&) const override {
        OPENVINO_THROW("NCS2: this plugin imports static MYRIAD 6.0 .blob only; compile IR using OpenVINO 2022.3 offline");
    }
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>&,
                  const ov::AnyMap&, const ov::SoPtr<ov::IRemoteContext>&) const override {
        OPENVINO_THROW("NCS2: remote contexts unsupported");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream, const ov::AnyMap& properties) const override {
        std::vector<uint8_t> bytes; char chunk[65536];
        while (stream) {
            stream.read(chunk, sizeof(chunk)); auto count = stream.gcount();
            require(bytes.size()+size_t(count) <= Blob::max_size, "blob too large");
            bytes.insert(bytes.end(), chunk, chunk+count);
        }
        require(stream.eof(), "failed to read blob");
        return load(std::move(bytes), properties);
    }
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream&, const ov::SoPtr<ov::IRemoteContext>&,
                                                    const ov::AnyMap&) const override {
        OPENVINO_THROW("NCS2: remote contexts unsupported");
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& tensor, const ov::AnyMap& properties) const override {
        require(tensor.is_continuous() && tensor.get_element_type() == ov::element::u8 &&
                tensor.get_byte_size() <= Blob::max_size, "expected compact u8 blob bytes");
        auto* data = tensor.data<const uint8_t>();
        return load(std::vector<uint8_t>(data, data+tensor.get_byte_size()), properties);
    }
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor&, const ov::SoPtr<ov::IRemoteContext>&,
                                                    const ov::AnyMap&) const override {
        OPENVINO_THROW("NCS2: remote contexts unsupported");
    }
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>&, const ov::AnyMap&) const override { return {}; }
    void set_property(const ov::AnyMap& values) override {
        std::lock_guard<std::mutex> lock(config_mutex);
        auto next = config; next.update(values); config = std::move(next);
    }
    ov::Any get_property(const std::string& name, const ov::AnyMap&) const override {
        if (name == "INTERNAL_SUPPORTED_PROPERTIES") return std::vector<ov::PropertyName>{};
        if (name == ov::supported_properties.name()) return std::vector<ov::PropertyName>{
            ov::supported_properties, ov::available_devices, ov::device::full_name, ov::device::capabilities,
            ov::device::id, {"NCS2_FIRMWARE_DIR", ov::PropertyMutability::RW}};
        if (name == ov::available_devices.name()) return devices();
        if (name == ov::device::full_name.name()) return std::string("Intel Neural Compute Stick 2 (MYRIAD blob runtime)");
        if (name == ov::device::capabilities.name()) return std::vector<std::string>{};
        std::lock_guard<std::mutex> lock(config_mutex);
        if (name == ov::device::id.name()) return config.id;
        if (name == "NCS2_FIRMWARE_DIR") return config.firmware;
        OPENVINO_THROW("NCS2: unsupported plugin property ", name);
    }
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap&) const override {
        OPENVINO_THROW("NCS2: remote contexts unsupported");
    }
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap&) const override {
        OPENVINO_THROW("NCS2: remote contexts unsupported");
    }
};
}  // namespace ncs2
#ifdef NCS2_TEST_TRANSPORT
static const ov::Version version = {"0.1.0-ov2026.0", "TEST ONLY: fake NCS2 USB transport"};
#else
static const ov::Version version = {"0.1.0-ov2026.0", "NCS2 native blob plugin"};
#endif
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ncs2::Plugin, version)
