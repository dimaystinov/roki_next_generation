# Модель для первого запуска NCS2

`ball.blob` действительно скомпилирована, не является заглушкой.
Источник: `Soccer/Vision/yolo_roma/orange_ball_on_green_only.{xml,bin}`
из коммита `ad372a887a6b62f51c2cba7a7156363dc642c882` репозитория Roki_2_Soccer.
Это найденный в истории кандидат для футбола. Актуальность весов и
распознавание на соревнованиях не проверены. Исторический код использует
класс 0 для мяча и 1 для корзины; исходный IR имеет выход [1,25200,7].

Собран нативный C++ OpenVINO 2022.3.2 (без Python). Его MYRIAD compiler
использован **только на машине подготовки модели**, с
`MYRIAD_ENABLE_MX_BOOT=NO`; устройство для компиляции не требовалось.
Программа подготовки — `openvino-ncs2/offline/compile_ball.cpp`.
В граф добавлены BGR U8 NHWC -> FP32 -> RGB -> деление на 255 и NCHW
для исходной сети; выход — FP32. Resize/letterbox выполняет Python-код.

Фактический результат `inspect_blob ball.blob`:

```
input images U8 shape=1,640,640,3, offset=0 bytes=1228800 compact=1
output output0 FP32 shape=1,25200,7, offset=0 bytes=705600 compact=1
```

Blob успешно импортирована новым OpenVINO 2026.0.0 и новым `Neural`
приложения через тестовый транспорт. Выполнение нейросети на NCS2
не проверено — физического устройства здесь нет.

## SHA256

```
88121e92ce20af2f2ae892c2b3128edd7d4f2aa20bc5f7ce2774dc3fd4ac4b83  ball.blob
44687cde2ea496f90c1c971bc07991658d6d50d9a2700c0358686dc90000260b  orange_ball_on_green_only.xml
1bbdcf47a18a6de16ff862dbee1bed5002c578dfb79844f247f8d00b0ba79a90  orange_ball_on_green_only.bin
47810e5ee29d974e308cce956668ecca498360cb5d2ab56c717932e635f9d524  usb-ma2x8x.mvcmd
```

Firmware получена upstream CMake из
`https://storage.openvinotoolkit.org/dependencies/myriad/firmware_usb-ma2x8x_20230418_40.zip`.
SHA256 архива из исходников upstream:
`bf61de4d01767ba85408eec3123c0038c5ead4712407dea12b04d822e5780c66`.
В комплекте именно USB-вариант firmware, не PCIe.

## Размещение в образе

- Пересобранный для CM4 `libopenvino_ncs2_plugin.so`: `/usr/lib/`.
- `ball.blob`: `/usr/share/roki/ball.blob`.
- `usb-ma2x8x.mvcmd`: например `/usr/share/roki/firmware/`.
- У процесса робота `NCS2_FIRMWARE_DIR=/usr/share/roki/firmware`.
- Новый OpenVINO **2026.0.0** и его bindings для Python образа, numpy, OpenCV,
  libusb и доступ к устройству USB.

Пути можно переопределить через `ROKI_NCS2_PLUGIN`, `ROKI_NCS2_BLOB`.
На голове не нужны OpenVINO 2022.3, старый Python или отдельный сервис.
Для защиты робота от нативных аварий приложение само запускает дочерний
процесс на том же новом Python. Библиотечный плагин от этого не меняется.
Перед запуском движений проверить `tools/test/check_openvino_ncs2.py`:

```sh
NCS2_FIRMWARE_DIR=/usr/share/roki/firmware \
python3 tools/test/check_openvino_ncs2.py \
  --plugin /usr/lib/libopenvino_ncs2_plugin.so \
  --model /usr/share/roki/ball.blob --iterations 100
```

Тест на нулях проверяет выполнение, не качество распознавания. Затем нужны
кадры поля, сравнение выходов с исходным стеком и длительный прогон NCS2.

## Повторная сборка blob

Проверенный host-конфиг OpenVINO 2022.3.2: `THREADING=SEQ`,
`ENABLE_INTEL_MYRIAD=ON`, `ENABLE_OV_IR_FRONTEND=ON`,
`ENABLE_COMPILE_TOOL=ON`. Отключены Python, samples, tests, CPU/GPU/GNA,
AUTO/AUTO_BATCH/MULTI/HETERO, ONNX/TF/Paddle frontends и LTO.
Для современного CMake передан `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
Подмодули: pugixml, ittapi, ade, json/nlohmann_json, gflags/gflags, xbyak.
Нужна libusb. Исходники старого OpenVINO при этой сборке не патчились.

После сборки targets `openvino_intel_myriad_plugin` и `openvino_ir_frontend`:

```sh
cmake -S openvino-ncs2/offline -B offline-build -DOpenVINO_DIR=/path/to/old-openvino-build
cmake --build offline-build
offline-build/compile_ball orange_ball_on_green_only.xml ball.blob
```
