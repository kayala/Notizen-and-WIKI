# Doxygen

Doxygen is an inline code tool for easy documentation.
There is also [standardese](https://github.com/standardese/standardese), which
aims to be a next generation of Doxygen.

## References

- [Doxygen website](http://www.doxygen.nl/)
- [IBM Developer](https://developer.ibm.com/technologies/systems/articles/au-learningdoxygen/)
- [Coding Style and Doxygen Documentation](https://www.cs.cmu.edu/~410/doc/doxygen.html)
- [Special commands](http://www.doxygen.nl/manual/commands.html)
- [Doxygen Quick
  Reference](http://www.mitk.org/images/1/1c/BugSquashingSeminars%242013-07-17-DoxyReference.pdf)

## installing \& using

```sh
git clone https://github.com/doxygen/doxygen.git
cd doxygen
mkdir build
cd build
cmake -G "Unix Makefiles" ..
make
sudo make install
```

If you want to have a frontend interface for edition the configuration file,
edit `CMakeLists.txt` at line `option(build_wizar "..." OFF)` to
`option(build_wizar "..." ON)` before `cmake -G "Unix Makefiles" ..`.

Inside the project folder, generate a configuration file with `doxygen -g
config_file_name`, edit `config_file_name` and then `doxygen config_file_name`
to generate html and latex (selectable).

Notes:
1. `config_file_name` is optional parameter
2. There are many things you can do with doxygen, the following is just the basic

Some fields are of special interest:

* `OUTPUT_DIRECTORY`: Where to place the documentation, default: project root
* `INPUT`: space-separated list of all the directories in which the sources \&
  headers reside
* `RECURSIVE`: specify if `INPUT` has subdirectories containig sources \&
  headers
* `FILE_PATTERNS`: different files extensions for sources \& headers
* `EXTRACT_ALL`: extract documentation even if file is undocumented (this might
  cause the comments to be phased)
* `SOURCE_BROWSER`: list of source files will be generated
* `INLINE_SOURCES` include the body of functions, classes and enums

Note: If coding in `C++`, graphs for `class`es are available.

## Darktheme

[Doxygen dark theme](https://github.com/MaJerle/doxygen-dark-theme)

## Syntax examples

### c

- `test.h`

```c
/** @file test.h
 *  \brief Declaration of test functions
 *
 *  @author Harry Q. Bovik (hqbovik)
 *  @author Fred Hacker (fhacker)
 *  @bug No known bugs
 */

#define __STDC_FORMAT_MACROS
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdbool.h>

/** example documentation */
#define SOMEDEF

/**
 * @defgroup TEST_GROUP Test Group
 * @{
 */

/** Test AAA documentation. */
#define TEST_AAA (1)
/** Test BBB documentation. */
#define TEST_BBB (2)
/** Test CCC documentation. */
#define TEST_CCC (3)

/** @} */

/** @struct mystruct
 *  @brief This structure is a test
 *  @var mystruct::a
 *  Member 'a' contains something
 *  @var mystruct::b
 *  Member 'b' contains some other thing
 */
struct mystruct
{
    int a;
    int b;
};

/** \brief Writes the current foreground and background
 *         color of characters printed on the console
 *         into the argument color.
 *  @param[in] a The address to which the current color
 *         information will be written.
 *  @param[out] b Arbitraty output parameter.
 *  @param[inout] c Second arbitraty input-output parameter.
 *  @param d Third arbitraty parameter.
 *  @return Functions return value
 *  @retval 1 when
 *  @retval 2 ok
 */
uint16_t funcao(int a, bool b, float c, int d);

/** @brief aleluia function.
 *  @return Praise God
 */
uint8_t aleluia(void);
```

- `test.c`

```c
/** @file test.c
 *  @brief Test functions
 *
 *  @author Harry Q. Bovik (hqbovik)
 *  @author Fred Hacker (fhacker)
 *  @bug No known bugs
 */

#include "test.h"

uint16_t funcao(int a, bool b, float c)
{
	return 0xff;
}

uint8_t aleluia(void) {

	char ftmp[64] = "1\0"; /// step comment
	/*sprintf(ftmp, "%d", 1);*/

	printf(ftmp); /// yet another comment

	return 0;
}
```

### VHDL

- Set `OPTIMIZE_OUTPUT_VHDL` to `YES`.
- An alternative to doxygen with VHDL is VHDocL (VHDLdoc)

```vhdl
-------------------------------------------------------
--! @file
--! @brief 2:1 Mux using with-select
-------------------------------------------------------

--! Use standard library
library ieee;
--! Use logic elements
	use ieee.std_logic_1164.all;

--! Mux entity brief description

--! Detailed description of this
--! mux design element.
entity mux_using_with is
	port (
		din_0	: in std_logic; --! Mux first input
		din_1	: in std_logic; --! Mux Second input
		sel		: in std_logic; --! Select input
		mux_out : out std_logic --! Mux output
	);
end entity;

--! @brief Architecture definition of the MUX
--! @details More details about this mux element.
architecture behavior of mux_using_with is
begin
	with (sel) select
		mux_out <= din_0 when '0',
		din_1 when others;
end architecture;
```

## Python

You can use either comments:

```python
##
# @file Beispiel.py

"""! @author vln
"""
```


## Special commands

- If \<sharp\> braces are used the argument is a single word.
- If (round) braces are used the argument extends until the end of the line on which the command was found.
- If {curly} braces are used the argument extends until the next paragraph. Paragraphs are delimited by a blank line or by a section indicator. Note that {curly} braces are also used for command options, here the braces are mandatory and just 'normal' characters. The starting curly brace has to directly follow the command, so without whitespace.
- If [square] braces are used the argument is optional.

|                |                  |                 |                |                  |                 |
| -------------- | ---------------- | --------------- | -------------- | ---------------- | --------------- |
| a              | f{               | remark          | dontinclude    | memberof         | tparam          |
| addindex       | f}               | remarks         | dot            | msc              | typedef         |
| addtogroup     | file             | result          | dotfile        | mscfile          | union           |
| anchor         | fn               | return          | e              | n                | until           |
| arg            | headerfile       | returns         | else           | name             | var             |
| attention      | hidecallergraph  | retval          | elseif         | namespace        | verbatim        |
| author         | hidecallgraph    | rtfinclude      | em             | nosubgrouping    | verbinclude     |
| authors        | hiderefby        | rtfonly         | emoji          | note             | version         |
| b              | hiderefs         | sa              | endcode        | overload         | vhdlflow        |
| brief          | hideinitializer  | secreflist      | endcond        | p                | warning         |
| bug            | htmlinclude      | section         | enddocbookonly | package          | weakgroup       |
| c              | htmlonly         | see             | enddot         | page             | xmlinclude      |
| callergraph    | idlexcept        | short           | endhtmlonly    | par              | xmlonly         |
| callgraph      | if               | showinitializer | endif          | paragraph        | xrefitem        |
| category       | ifnot            | showrefby       | endinternal    | param            | $               |
| cite           | image            | showrefs        | endlatexonly   | parblock         | @               |
| class          | implements       | since           | endlink        | post             | \               |
| code           | include          | skip            | endmanonly     | pre              | &               |
| cond           | includedoc       | skipline        | endmsc         | private          | ~               |
| copybrief      | includelineno    | snippet         | endparblock    | privatesection   | <               |
| copydetails    | ingroup          | snippetdoc      | endrtfonly     | property         | =               |
| copydoc        | internal         | snippetlineno   | endsecreflist  | protected        | >               |
| copyright      | invariant        | startuml        | endverbatim    | protectedsection | #               |
| date           | interface        | struct          | enduml         | protocol         | %               |
| def            | latexinclude     | subpage         | endxmlonly     | public           | "               |
| defgroup       | latexonly        | subsection      | enum           | publicsection    | .               |
| deprecated     | li               | subsubsection   | example        | pure             | ::              |
| details        | line             | tableofcontents | exception      | ref              | |               |
| diafile        | link             | test            | extends        | refitem          | --              |
| dir            | mainpage         | throw           | f$             | related          | ---             |
| docbookinclude | maninclude       | throws          | f[             | relates          |                 |
| docbookonly    | manonly          | todo            | f]             | relatedalso      |                 |

## Adding images

Use the `@image` comand to add an image. HTML has been tested to use `svg`, `png` and `jpg`.
Syntax is as follows:

```c
@image html people.svg
@image latex people.eps "People image" width=\textwidth
```

## Equations

Hint: enable the `USE_MATHJAX` flag.

There are three mathmodes in doxygen:

- `@f$` inline math
- `@f[` normal math
- `@f{` environment math

Examples:

```c
The distance between \f$(x_1,y_1)\f$ and \f$(x_2,y_2)\f$ is \f$\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}\f$.
\f[
  |I_2|=\left| \int_{0}^T \psi(t)
           \left\{
              u(a,t)-
              \int_{\gamma(t)}^a
              \frac{d\theta}{k(\theta,t)}
              \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
           \right\} dt
        \right|
\f]
\f{eqnarray*}{
     g &=& \frac{Gm_2}{r^2} \\
       &=& \frac{(6.673 \times 10^{-11}\,\mbox{m}^3\,\mbox{kg}^{-1}\,
           \mbox{s}^{-2})(5.9736 \times 10^{24}\,\mbox{kg})}{(6371.01\,\mbox{km})^2} \\
       &=& 9.82066032\,\mbox{m/s}^2
\f}
```

Note: You can also do newcommands and include libraries as in LaTeX.

## Mainpage

The mainpage is not automatically generated. You can set it with `@mainpage`
command or indicate a markdown page to use. To reference this markdown page, you
should use the flag `USE_MDFILE_AS_MAINPAGE` while also including the referenced
file in the `INPUT` list.
