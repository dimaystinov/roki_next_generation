# NCS2 — плагин для OpenVINO 2026.0

Реализован импорт статических MYRIAD `.blob` формата 6.0 и выполнение через
`mvnc → XLink → libusb → NCS2`. Новый `openvino.Core` и новый Python работают
в одном процессе инференса. Приложение робота запускает его как приватный
дочерний процесс на том же новом Python, чтобы переживать сбои NCS2.
Старый OpenVINO runtime, его Python bindings и сервис на старом Python
не используются. Из старого дерева собирается только транспорт.

**Статус:** библиотека собрана и загружается настоящим OpenVINO 2026.0.0 на
Python 3.14.3 / macOS arm64. Интеграционные тесты с имитацией USB проходят.
USB boot и inference на физическом NCS2, а также сборка под Buildroot CM4
ещё не проверены: устройство и toolchain не предоставлены. Историческая
модель мяча найдена, скомпилирована в `../ball.blob`; её физический I/O
контракт совпал с новым Python-кодом. См. `../MODEL_BUILD_RU.md`.
Этот статус не следует понимать как подтверждение работы нейросети робота.

## Зафиксированные зависимости

| Компонент | Версия / commit |
|---|---|
| OpenVINO runtime и соответствующие developer headers | 2026.0.0 / `c6d6a13a8863f576b627dabcda0c40b090638f9e` |
| Исходники mvnc/XLink | OpenVINO 2022.3.2 / `e2c7e4d7b4d6b315c0b62a22438146b28d15f1ee` |
| USB firmware исходной ветки | `usb-ma2x8x.mvcmd`, пакет `20230418_40` |
| Локальная проверка libusb | 1.0.30 |

Исходники берутся из https://github.com/openvinotoolkit/openvino.
Выпуск нового runtime и developer headers должны совпадать: plugin API —
внутренний C++ API. Другие выпуски требуют пересборки и проверки интерфейса.
Исходники плагина включены в `buildroot/openvino-ncs2/` этого репозитория.
Для образа их нужно собрать инструментами и sysroot соответствующего Buildroot.

## Сборка

Указать **абсолютные пути** к исходникам двух версий и установленному
OpenVINO 2026.0.0. В source нового выпуска нужны каталоги
`src/{inference,core}/dev_api`, `src/common/{util,itt}/include`.
В старом — `src/plugins/intel_myriad/third_party` (mvnc и XLink).
Нужны CMake, C/C++17 compiler, Threads и libusb development files.
Сборка не скачивает зависимости из сети.

```sh
cmake -S . -B build \
  -DOpenVINO_DIR=/path/to/openvino/cmake \
  -DNCS2_OPENVINO_SOURCE_DIR=/path/to/openvino-2026.0.0 \
  -DNCS2_LEGACY_SOURCE_DIR=/path/to/openvino-2022.3.2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF
cmake --build build --parallel 4
```

Для Buildroot дополнительно передать его `CMAKE_TOOLCHAIN_FILE`; OpenVINO,
libusb и pkg-config должны разрешаться в staging sysroot для CM4, не хоста.
Установить только `libopenvino_ncs2_plugin.so` в целевой libdir через
`DESTDIR=... cmake --install build --prefix /usr`. Firmware поставить отдельно
и указать каталог через `NCS2_FIRMWARE_DIR` или одноимённое свойство плагина.
Не устанавливать `libopenvino_ncs2_test_plugin`: это тестовая имитация USB,
она намеренно не входит в install target.

Сохранить доступ к USB-устройству до и после загрузки firmware. В исходном
mvnc есть `src/97-myriad-usbboot.rules`; правила доступа адаптирует образ.

## Вызов из нового Python

```python
import io
from pathlib import Path
import numpy as np
import openvino as ov

core = ov.Core()
core.register_plugin('/usr/lib/libopenvino_ncs2_plugin.so', 'MYRIAD')
compiled = core.import_model(
    io.BytesIO(Path('/path/to/ball.blob').read_bytes()),
    'MYRIAD', {'NCS2_FIRMWARE_DIR': '/path/to/firmware'},
)
for port in compiled.inputs:
    print(port.any_name, port.shape, port.element_type)
request = compiled.create_infer_request()
# Только smoke test. Для распознавания здесь нужны подготовленные данные камеры.
inputs = {i: np.zeros(port.shape, dtype=port.element_type.to_dtype())
          for i, port in enumerate(compiled.inputs)}
request.infer(inputs)
outputs = [request.get_output_tensor(i).data.copy()
           for i in range(len(compiled.outputs))]
```

`set_tensor` / `start_async` / `wait` / `AsyncInferQueue` работают через
стандартный OpenVINO API. Передача в общий FIFO сериализована, поэтому ответы
не смешиваются между запросами. После ошибки передачи Session помечается
неисправной: нужно освободить старые requests/compiled model и импортировать
модель заново. Таймаут транспорта задан 10 секунд, watchdog — 1 секунда.
Количество одновременно открытых моделей/устройств на железе не проверено;
первый целевой сценарий — один NCS2 и одна модель.

## Контракт тензоров — существенно для интеграции

Плагин показывает **физические I/O тензоры из blob**, в порядке записи
портов, с внешними осями первыми. Если blob хранит NHWC, вход будет NHWC;
если NCHW — NCHW. Типы: FP16, U8, I32, FP32, I8. Padding и byte strides
разбираются из blob; компактные тензоры копируются одним memcpy, padded —
с учётом stride. Между packed FIFO и OpenVINO тензорами есть копирование.

Автоматических resize, BGR↔RGB, FP32↔FP16 и нормализации **нет**. Дописанные
старым exporter сведения об исходном логическом IR-интерфейсе не применяются.
Поэтому интерфейс может отличаться от старого Python worker, который делал
preprocessing. Для BGR U8 прямо с камеры модель должна быть заранее
скомпилирована с подходящими U8-входом и layout. Иначе preprocessing надо
реализовать согласно рабочей модели, а не угадывать по её размерам.

Много входов/выходов поддерживается как несколько областей одного packed
FIFO; перекрывающиеся области, dynamic shape (`@shape`), пустые/слишком
большие тензоры и другие форматы blob отклоняются явно.
Непрерывность пользовательского host tensor проверяется; remote tensors
и batch через `set_tensors` не заявлены.

В приложении `Neural` уже заменён локальным вызовом этого плагина.
Сохраняется прежний возврат центра мяча `(cx, cy)`, но исправлены формат
xywh для NMS, выбор самой уверенной детекции и пересчёт padding/scale.
При недоступности backend он сообщает ошибку и `is_ready=False`;
существующая футбольная стратегия может перейти к цветовому детектору.

`compile_model(XML/ONNX)` пока намеренно возвращает ошибку. Для этой версии
нужна готовая blob от OpenVINO 2022.3; её можно получить заранее на машине
подготовки моделей. Это не старый Python-воркер на голове. CPU fallback нет.
`get_runtime_model` представляет непрозрачный граф NCS2 с I/O-метаданными,
а не восстановленную структуру исходной нейросети. Profiling не реализован.

## Проверки

Для локальных тестов добавить `-DBUILD_TESTING=ON` и
`-DNCS2_TEST_PYTHON=/path/to/python-with-openvino`, затем:

```sh
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

- Parser: порядок осей, padding, compact copy и отказ на повреждённых blob.
- Настоящие Python/OpenVINO + тестовая USB-библиотека: импорт/экспорт,
  синхронные и асинхронные запросы, четыре конкурентных requests,
  соответствие ответов входам и блокировка дальнейшего inference после timeout.
- Настоящая библиотека с mvnc/XLink: `core.get_versions('MYRIAD')` возвращает
  `NCS2 native blob plugin`; на тестовом Mac устройств нет (`[]`).

Для головы использовать `tools/test/check_openvino_ncs2.py` из приложения
с `--plugin`, `--model` и `--inputs` (NPZ с input_0, input_1, ...).
Затем сравнить выходы с рабочим старым стеком на одних данных и проверить
USB disconnect/reconnect, перезапуск, длительную работу и фактическую задержку.
Без этого аппаратная совместимость остаётся непроверенной.
