# Hardware

- [mbedded.ninja](https://blog.mbedded.ninja/)

- simplex: data is sent in one direction only
- half-duplex: only one direction at a time, but can be sent both ways
- full-duplex: can send to both directions simultaneously

## Endianess

Memo:
Example: 0x0102

- Big Endian: the MSB (big) end comes first: 0x01, 0x02
- Little Endian: the LSB (little) end comes first: 0x02, 0x01

[Image with description](https://en.wikipedia.org/wiki/File:Endianessmap.svg)

The web is always Big-endian (covention). For programming in C, there are standard libraries to
convert from internet convention to host machine: see `htonl` (host to long).

## PCIe bus

[PCIe bus page](pcie.md)

## SPI bus

- [Using Serial Peripheral Interface (SPI) Master and Slave with Atmel AVR
  Microcontroller ](http://www.ermicro.com/blog/?p=1050)
- [Introduction to SPI
  interface](https://www.analog.com/media/en/analog-dialogue/volume-52/number-3/introduction-to-spi-interface.pdf)

- MISO = Master Input, Slave Output
- MOSI = Master Output, Slave Input

### naming conventions

- [SPI bus: Clock Polarity and Clock Phase by
  example](https://deardevices.com/2020/05/31/spi-cpol-cpha)

- SDI = Serial Data Input
- SDO = Serial Data Output
- DI = Data Input
- DO = Data Output
- SI = Serial Input
- SO = Serial Output
- CPOL = Clock POLarity (rising oder falling edge)
  - SCL is high when no transmission is occurring means '1' (active high)
- CPHA = Clock PHase
  - data sampling, 1 = sample on the second clock edge; first determine CPOL

MOSI:

- SDI, DI, DIN, SI - on slave devices; connects to MOSI on master
- SDO, DO, DOUT, SO - on master devices; connects to MOSI on slave

MISO:

- SDO, DO, DOUT, SO - on slave devices; connects to MISO on master
- SDI, DI, DIN, SI - on master devices; connects to MISO on slave

```
┌───────────────┐    ┌───────────────┐
│               │    │               │
│            CSn├───►│CSn            │
│  MASTER       │    │        SLAVE  │
│            SCL├───►│SCL            │
│               │    │               │
│           MOSI├───►│SDI            │
│               │    │               │
│           MISO│◄───┤SDO            │
│               │    │               │
└───────────────┘    └───────────────┘
```

Devices other than MCUs having SPI interface tend to use SDI/SDO or DIN/DOUT
convention (e.g. ADC/DAC chips, digital potentiometers, sensors etc). But MCUs
always use MOSI/MISO convention (MSP430 series MCUs have a slight difference at
this point: They have SIMO/SOMI pins which are totally the same as MOSI/MISO).
This is because the MCUs are assumed (which should be) to be master.

With 3-wire interface, there is only: CS, SCL and Data. Data bus is negotiated
between master and slave.

### two MCUs communicating

If two MCUs are communicating through SPI then the "master" tag alternates
between the two (i.e. sender becomes "master" at the time), but the connection
does not change: MOSI on the 1st chip connects to MISO on the 2nd chip and
vice-versa.

### length matching

Length matching is not necessary.

### implementation

Master and slave hardwares are almost the same, the only difference is where the
clock generation lies.

SPI can be describes as a shared circular buffer make out of shift registers.

```
  ┌───────────────────┐      ┌───────────────────┐
  │      MASTER       │      │      SLAVE        │
  │ ┌─┬─┬─┬─┬─┬─┬─┬─┐ │ MOSI │ ┌─┬─┬─┬─┬─┬─┬─┬─┐ │
┌─┤►│0│1│2│3│4│5│6│7├─┼──────┤►│0│1│2│3│4│5│6│7├─┼─┐
│ │ └─┴─┴─┴─┴─┴─┴─┴─┘ │ SCL  │ └─┴─┴─┴─┴─┴─┴─┴─┘ │ │
│ │                   ├─────►│                   │ │
│ └───────────────────┘      └───────────────────┘ │
│                       MISO                       │
└──────────────────────────────────────────────────┘
```

### modes

The following table is a standard, but it cab also vary - always check:

mode|CPOL|CPHA
-|-|-
0|0|0
1|0|1
2|1|0
3|1|1

## I2C

- [I2C Bus Specification](https://i2c.info/i2c-bus-specification)
- [I²C-Bus Specification and User Manual](https://www.nxp.com/docs/en/user-guide/UM10204.pdf)

### Reserved addresses

slave adr.|rd|description
-|-|-
0000 000|0|general call address
0000 000|1|START byte
0000 001|x|CBUS address
0000 010|x|Reserved for different bus format
0000 011|x|Reserved for future purposes
0000 1xx|x|Hs-mode master code
1111 1xx|x|Reserved for future purposes
1111 0xx|x|10b slave addressing

### standard values

- read/write bit: read = 1, write = 0

### data frame

```
[start][address][wr/rd][ack/nack][data 1][ack/nack][data 2][ack/nack]...[stop]
```

- start: 1 bit
- address: 7 to 10 bits
- wr/rd: 1 bit
- ack/nack: 1 bit
- data: 8 bits
- stop: 1 bit

### length matching

Length matching is not necessary.

## UART USART RS232

- [UART: A Hardware Communication Protocol Understanding Universal Asynchronous
  Receiver/Transmitter](https://www.analog.com/en/analog-dialogue/articles/uart-a-hardware-communication-protocol.html)

### pins and glossary

- XOFF, XON: special commands to start/stop slave
- DTR: Data Terminal Ready, line to say that there is a receiver on the other
  side, connect to DSR
- DSR: Data Set Ready
- RTS: Request To Send, coordinates when to send the data, connect to CTS
- CTS: Clear To Send
- Tx and Rx are normally high
- DCD: Data Carrier Detect, outdated, was used by modems, tells when an analog
  signal was being received
- RI: Ring Indicator, outdates, was used by modems, tells that the phone is
  ringing
- SG: Signal Ground, same as GND

RTS and CTS and known as "hardware flow control", whereas XOFF and XON are known
as "software flow control".

- If DTR and CTS are not used, just connect both from the same end together.
- If RTS and CTS are not used, just connect both from the same end together.

## packet frame

```
[start bit][data frame][parity bit][stop bits]
```

- start bit: 1 bit
- data frame: between 5 to 9 bits
- parity bit: 1 bit wide, optional
- stop bits: 1 to 2 bits, second bit may indicate next package frame separation

### standard values

- Baud rate: 9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600, 1000000,
  1500000 [bps]
- data frame is usually LSB-first
- 1 start bit
- 8 data bits
- 1 parity bit
- 1 stop bit

## MCU
- [Debugging ARM based microcontrollers in neovim with
  gdb](https://chmanie.com/post/2020/07/18/debugging-arm-based-microcontrollers-in-neovim-with-gdb/)
- [Bare Metal STM32
  Programming](https://vivonomicon.com/2018/04/02/bare-metal-stm32-programming-part-1-hello-arm/)
- [GCC arm
  cross-compiler](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads)

General information about MCUs.

Note: for Linux serial, use `minicom`.

## STM32

There are different ways to achieve this.

- [Platformio libre, multiple vendors compilation tool](platformio.org)
- [Start Developing STM32 on
  Linux](https://www.instructables.com/Start-Developing-STM32-on-Linux/)
- [Learning STM32](https://riptutorial.com/ebook/stm32)

- STM32Nucleo: Device
- STM32Cube: IDE suite (not necessary)
- STM32CubeMX: Code generator
- STLink-V2: Hardware programmer
- STLink: Firmware programmer
- [OpenOCD](https://sourceforge.net/p/openocd/code/ci/master/tree/): Firmware
  debugger and programmer
- SWD: Serial Wire Debug
- [libopemcm3](https://github.com/libopencm3/): Low level libre library for ARM
  cortex
- newlib: _one_ C standard library implementation for embedded devices

Notes:

- It is recommended to compile Open On-Chip Debugger from source
- STLink has a GUI (`stlink-gui`)
- STLink-V2 can be replaced with a USB-serail converter (like the FT232,
  CP2102...), but it gives you debugging options
- STM32CubeIDE can generate Makefile projects

### arm coss-compiler

- install gcc and newlib: `pacman -S arm-none-eabi-gcc arm-none-eabi-newlib`
- optionally install utilities: `pacman -S arm-none-eabi-binutils`: utilities
  like objcpy, objdump, strip, readelf, ar, as, ld, etc

### flashing with ST-LINK

- [libopemcm3 examples](https://github.com/libopencm3/libopencm3-examples)
- [Linux STM32 Discovery
  GCC](https://www.wolinlabs.com/blog/linux.stm32.discovery.gcc.html)
- [STM32 Boitier
  ST-LINK](https://riton-duino.blogspot.com/2019/03/stm32-boitier-st-link.html)

One way to flash the device is:

- Add the following udev rule:

```
# /etc/udev/rules.d/45-usb-stlink-v2.rules

#FT232
ATTRS{idProduct}=="6014", ATTRS{idVendor}=="0403", MODE="666", GROUP="plugdev"

#FT2232
ATTRS{idProduct}=="6010", ATTRS{idVendor}=="0403", MODE="666", GROUP="plugdev"

#FT230X
ATTRS{idProduct}=="6015", ATTRS{idVendor}=="0403", MODE="666", GROUP="plugdev"

#STLINK V1
ATTRS{idProduct}=="3744", ATTRS{idVendor}=="0483", MODE="666", GROUP="plugdev"

#STLINK V2
ATTRS{idProduct}=="3748", ATTRS{idVendor}=="0483", MODE="666", GROUP="plugdev"

#STLINK V2.1
ATTRS{idProduct}=="374b", ATTRS{idVendor}=="0483", MODE="666", GROUP="plugdev"
```

Note: The user should belong to the `GROUP` named above.

- connect the device through your STLink-V2 (or integrated ST-LINK hw)
- probe to see if the board is correctly connected: `st-info --probe`
- erase the MCU: `st-flash erase`

Write a program with `libopencm3` library as a substitute for HAL. Create a
linker script defining
the ROM and RAM.

- compile to object: `arm-none-eabi-gcc -mcpu=cortex-m3 -mthumb -Wall -g -O0 -I
  . -I lib/inc -c -o
  main.o main.c`
- link to ELF: `arm-none-eabi-gcc
  -Wl,--gc-sections,-Map=main.elf.map,-cref,-u,Reset_Handler -I . -I
  lib/inc -L lib -T stm32.ld main.o stm32f10x_it.o lib/libstm32.a --output
  main.elf`
- link to bin: `arm-none-eabi-objcopy -O binary main.elf main.bin`
- program the firmware: `st-flash write main.bin 0x08000000` (this number
  corresponds to a region
  configured by the linker as ROM)
- STM32CubeMX can create Makefile

### flashing with OpenOCD

- [Flashing and debugging STM32 microcontrollers under
  Linux](https://cycling-touring.net/2018/12/flashing-and-debugging-stm32-microcontrollers-under-linux)
- [STM OpenOCD fork](https://github.com/STMicroelectronics/OpenOCD): this may
  have been merged to the official later on.

- install telnet
- copy the udev rule inside `/usr/share/openocd/contrib` to the udev
- search for your config file under `/usr/share/openocd/script`, there are
  config files for MCU, programmers - and board, which combine both
- use `openocd -f config1.cfg -f ...`: this opens a telnet (programming) and tcl
  (debugging) connections
- `telnet localhost <port>`: to get the programming interface in another
  terminal
- type:

```
init
reset init
halt
flash write_image erase myprogram.elf
exit
```

#### debugging with OpenOCD

Do the above to open a tcl connection and then:

- `arm-none-eabi-gdb myprogram.elf
- `(gdb) target extended-remote localhost:3333` (assuming 3333 is the tcl port)
- `(gdb) monitor reset halt`
- `(gdb) load`

### troubleshooting

- After installation, try running `dmesg` and see if there is an error message
  related to the STM. If there is an error message:
  - test if you can run `st-link` and `st-util`
  - try changing USB ports
  - try updating the STLink through the IDE
- [STM32 Standard Periferal
  Library](https://www.st.com/en/embedded-software/stm32-standard-peripheral-libraries.html):
  some MCUs are not included
- If MCU is not included, then maybe try the STM32CubeMX

### creating the Makefile from STM32CubeMX

- [Makefile4CubeMX](https://github.com/duro80/Makefile4CubeMX)
- [Hacky Makefile generator](https://github.com/baoshi/CubeMX2Makefile)
