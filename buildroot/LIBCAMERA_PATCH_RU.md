# Для агента, который собирает образ CM4

Нужно добавить в образ один патч libcamera, чтобы приложение получало готовый
BGR-кадр и аппаратный номер Unicam без дополнительного RAW-потока.

**База:** `raspberrypi/libcamera`, тег `v0.7.2+rpt20260817`, commit
`6c1dd9d55573010f710c9e190a73e7e76f0d9432`.

Патч: `0001-rpi-unicam-sequence.patch` рядом с этим файлом.

## Что делает патч

1. В конец `src/libcamera/control_ids_rpi.yaml` добавляется выходное поле
   `UnicamSequence`, тип `int64_t`. Оно добавлено в конец, чтобы не сдвигать
   существующие ID этого vendor namespace в немодифицированной базе.
2. В `Vc4CameraData::tryRunPipeline()` в `src/libcamera/pipeline/rpi/vc4/vc4.cpp`
   номер берётся из `bayerFrame.buffer->metadata().sequence` и записывается в
   `request->_d()->metadata()` после `fillRequestMetadata`, до запуска ISP.
3. Python bindings генерируют `libcamera.controls.rpi.UnicamSequence` из YAML.
   Ручные изменения в сгенерированных файлах не нужны.

Номер относится к Bayer-буферу, который действительно обрабатывается для
текущего запроса. Здесь нельзя использовать номер request, номер ISP-буфера,
временную метку или оценку по FPS. RAW-поток приложения не создаётся; пиксели
этим патчем не копируются и не преобразуются.

## Интеграция в сборку

Добавьте патч в существующую очередь патчей пакета libcamera в системе сборки
образа. Сохраните свои патчи удаления ненужных сенсоров/механизмов.

В исходниках libcamera проверка и применение выглядят так; замените путь на
реальное расположение приложенного файла:

```sh
git apply --check /path/to/0001-rpi-unicam-sequence.patch
git apply /path/to/0001-rpi-unicam-sequence.patch
```

Пересоберите и установите согласованный комплект libcamera, RPi IPA и Python
bindings. В Meson Python bindings должны оставаться включены:
`-Dpycamera=enabled`. Не смешивайте старые сгенерированные control IDs и новые
библиотеки. Если ваши патчи удаляют controls из YAML, проверьте согласованность
всех пересобранных компонентов; числовой ID в приложении не захардкожен.

Проверка установленного Python API:

```sh
python3 -c 'import libcamera; print(libcamera.controls.rpi.UnicamSequence)'
```

Далее запустите из корня Roki_2_Soccer:

```sh
python3 tools/test/check_libcamera.py --frames 100 --restarts 2
```

Ожидается BGR `uint8`, форма `(650, 800, 3)`, реальные номера с сохранением
пропусков. После перезапуска начинается новая эпоха счётчика; первый доступный
кадр не обязан иметь номер 0. Этот скрипт проверяет только камеру и не открывает
аппаратный канал Roki. Совпадение с STM STROBE/IMU нужно затем проверить на роботе.

## Проверено при подготовке

- Патч проверен на точном исходном теге.
- Генераторы C++ control IDs и Python controls обработали изменённый YAML.
- В результате есть `Control<int64_t> UnicamSequence` и Python-экспорт
  `rpi.UnicamSequence`.
- Адаптер и передача номера в IMU проверены тестами с подменой оборудования.

Полная сборка C++ для CM4, установка образа и аппаратный захват здесь не
выполнялись. Патч ограничен VC4/CM4 и не добавляет поле в PiSP-пайплайн Pi 5.

## Проверенные исходники

- [Точный тег libcamera](https://github.com/raspberrypi/libcamera/tree/v0.7.2%2Brpt20260817)
- [VC4 pipeline](https://github.com/raspberrypi/libcamera/blob/v0.7.2%2Brpt20260817/src/libcamera/pipeline/rpi/vc4/vc4.cpp)
- [Python bindings](https://github.com/raspberrypi/libcamera/blob/v0.7.2%2Brpt20260817/src/py/libcamera/py_main.cpp)
- [Форматы и V4L2 mapping](https://github.com/raspberrypi/libcamera/blob/v0.7.2%2Brpt20260817/src/libcamera/formats.cpp)
