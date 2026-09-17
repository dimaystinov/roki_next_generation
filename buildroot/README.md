# Комплект для сборщика образа Roki / CM4

Здесь находятся патчи и артефакты, которые нужно включить в **существующий
Buildroot головы**. Разработчик образа опубликует его код и релиз после
собственных испытаний. Этот каталог не содержит defconfig, ядро, board files,
загрузчик, готовый SD/eMMC image или firmware STM. Поэтому он сам по себе
не собирает и не прошивает всю голову.

## Состав

| Файл / каталог | Назначение |
|---|---|
| `0001-rpi-unicam-sequence.patch` | Дополнение libcamera VC4: аппаратный номер Unicam в metadata |
| [LIBCAMERA_PATCH_RU.md](LIBCAMERA_PATCH_RU.md) | Точная база патча, применение, пересборка и проверка |
| [openvino-ncs2/](openvino-ncs2/README_RU.md) | Исходники C++ плагина MYRIAD, CMake, parser/USB tests, offline compiler |
| [OPENVINO_NCS2_RU.md](OPENVINO_NCS2_RU.md) | Состояние реализации и требования к образу |
| [MODEL_BUILD_RU.md](MODEL_BUILD_RU.md) | Источник модели, воспроизводимая подготовка и размещение |
| `orange_ball_on_green_only.xml`, `.bin` | Исторический исходный IR и веса кандидата для футбола |
| `ball.blob` | Реальный статический NCS2 blob, подготовленный из этого IR |
| `usb-ma2x8x.mvcmd` | USB firmware для Myriad X, загружаемая в NCS2 при открытии |
| `SHA256SUMS` | Контроль целостности патча, модели и firmware |

## 1. libcamera

База: [raspberrypi/libcamera](https://github.com/raspberrypi/libcamera),
`v0.7.2+rpt20260817`, commit `6c1dd9d55573010f710c9e190a73e7e76f0d9432`.
Включить приложенный patch в очередь существующего пакета Buildroot,
сохранив патчи платформы, удаляющие ненужные сенсоры/механизмы.

Проверка в checkout точной базы перед интеграцией:

```sh
git apply --check /path/to/roki_next_generation/buildroot/0001-rpi-unicam-sequence.patch
```

Пересобрать библиотеку, RPi IPA и Python bindings вместе. В Meson требуется
`-Dpycamera=enabled`. Контроль, который должен появиться у установленного Python:

```sh
python3 -c 'import libcamera; print(libcamera.controls.rpi.UnicamSequence)'
```

Патч предназначен для VC4/CM4; PiSP на Pi 5 не поддерживается этой правкой.
Ручной RAW-поток приложения для него не нужен.

## 2. OpenVINO и нативный NCS2-плагин

**Основной вариант:** [форк OpenVINO](https://github.com/dimaystinov/openvino)
на базе 2026.5.0. В нём есть исходники транспорта и плагина, переключатель
`ENABLE_INTEL_MYRIAD=ON`, штатная регистрация и упаковка MYRIAD.
Собирать runtime, плагин и Python bindings из одного checkout форка.
См. [README_NCS2_RU.md](https://github.com/dimaystinov/openvino/blob/master/README_NCS2_RU.md).
Модель, firmware и патч libcamera брать из этого каталога.

**Альтернатива — прежний standalone-комплект:** следующие команды и каталог
`openvino-ncs2/` относятся строго к OpenVINO 2026.0.0. Они сохранены как
проверенная исходная реализация; не собирать этот плагин против 2026.5.

Runtime и headers: OpenVINO **2026.0.0**, commit
`c6d6a13a8863f576b627dabcda0c40b090638f9e`.
Исходники транспорта: OpenVINO **2022.3.2**, commit
`e2c7e4d7b4d6b315c0b62a22438146b28d15f1ee`.
Из старого дерева линкуются только mvnc/XLink; старый runtime в образ не нужен.
Полное upstream дерево OpenVINO здесь не дублируется; точные исходники
должна предоставить сборка. Плагин собирается отдельно, это не `.patch`
поверх современных исходников OpenVINO.

Пример команды конфигурации; пути заменить на реальные пути сборщика:

```sh
cmake -S buildroot/openvino-ncs2 -B /path/to/ncs2-build \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/buildroot/output/host/share/buildroot/toolchainfile.cmake \
  -DOpenVINO_DIR=/path/to/target-openvino/cmake \
  -DNCS2_OPENVINO_SOURCE_DIR=/path/to/openvino-2026.0.0 \
  -DNCS2_LEGACY_SOURCE_DIR=/path/to/openvino-2022.3.2 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build /path/to/ncs2-build --parallel 4
DESTDIR=/path/to/target-root cmake --install /path/to/ncs2-build --prefix /usr
```

libusb, Threads, OpenVINO и pkg-config должны разрешаться в **target sysroot**.
Установить production `libopenvino_ncs2_plugin.so`; тестовый
`libopenvino_ncs2_test_plugin` имитирует USB и не должен попадать в образ.
Готовых `.so` с машины разработчика в этом репозитории нет: нужен целевой ABI.

USB-доступ нужен как до, так и после загрузки firmware. Исходный mvnc содержит
`97-myriad-usbboot.rules`; применить эквивалентные правила в используемой
системе устройств образа. Встраивание в конкретный `.mk`/`Config.in`,
зависимости пакетов и post-build hooks задаёт существующий Buildroot проекта.

Подробности API, типов тензоров, ограничений и тестов:
[openvino-ncs2/README_RU.md](openvino-ncs2/README_RU.md).

## 3. Модель, firmware и приложение

Пример наполнения уже подготовленного target root, не работающего диска:

```sh
install -Dm644 buildroot/ball.blob /path/to/target-root/usr/share/roki/ball.blob
install -Dm644 buildroot/usb-ma2x8x.mvcmd /path/to/target-root/usr/share/roki/firmware/usb-ma2x8x.mvcmd
```

В окружении запуска приложения:

```sh
export ROKI_NCS2_PLUGIN=/usr/lib/libopenvino_ncs2_plugin.so
export ROKI_NCS2_BLOB=/usr/share/roki/ball.blob
export NCS2_FIRMWARE_DIR=/usr/share/roki/firmware
export ROKI_NCS2_TIMEOUT=1
export ROKI_NCS2_STARTUP_TIMEOUT=15
```

Установить Python образа, его OpenVINO bindings, numpy и OpenCV. Дочерний
нейропроцесс запускается автоматически через тот же интерпретатор. Отдельный
сервис старого Python не нужен. Python должен поддерживать используемый POSIX
API, включая subprocess pass_fds и socketpair; локально проверен Python 3.14.3.
Для робота нужны также аппаратный модуль `Roki`, локальные калибровки и
зависимости выбранных режимов, перечисленные в [корневом README](../README.md).

Код можно разместить в `/opt/roki`; init/system service в этом комплекте не
добавлен, его связывает с существующим запуском разработчик образа. Не включать
в runtime дерево `.git`, `tests`, исходники библиотек и host build directories.
XML/BIN нужны на машине подготовки модели; для запуска достаточно `.blob`.

## 4. Проверка комплекта

Из этого каталога на Linux:

```sh
sha256sum -c SHA256SUMS
```

На macOS эквивалент — `shasum -a 256 -c SHA256SUMS`.
Патч проверяется на указанном теге, C++ тесты — по инструкции плагина,
Python-тесты — `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests`
из корня приложения.

На CM4 последовательно проверить libcamera, затем NCS2 диагностическими
скриптами из корневого README. После этого проверить USB disconnect во время
работы `Neural`, цветовой fallback и продолжение камеры/STM. Выставить timeout
по измерениям реальной модели. До испытаний не считать прошиваемый образ
готовым к автономному матчу: отдельно остаются ошибки локализации.

## Происхождение файлов

Модель и веса восстановлены из истории исходного
[Roki_2_Soccer](https://github.com/aleksey-yagubov/Roki_2_Soccer), commit
`ad372a887a6b62f51c2cba7a7156363dc642c882`.
USB firmware — исходный пакет OpenVINO, URL и SHA256 записаны в
[MODEL_BUILD_RU.md](MODEL_BUILD_RU.md).
Для исходников плагина приложен [Apache-2.0 LICENSE](openvino-ncs2/LICENSE).
Этот файл не меняет лицензионные условия исходного приложения, модели,
firmware или других upstream компонентов.
