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
