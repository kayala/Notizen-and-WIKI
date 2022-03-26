# LaTeX

## References

- [bibtex vs. biber and biblatex vs. natbib](https://tex.stackexchange.com/questions/25701/bibtex-vs-biber-and-biblatex-vs-natbib)
- [Bibtex, Latex compiling](https://tex.stackexchange.com/questions/204291/bibtex-latex-compiing#204298)
- [Using Latexmk](https://mg.readthedocs.io/latexmk.html)

## Different languages

- `inputenc` allows to input foreign characters directly to the document
- `fontenc` is the font encoding (using this may be unnecessary, since the
  default encoding works for most cases - but it's a good idea to avoid glitches)
- `babel` for proper hyphenation and translating names of document elements

```tex
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[ngerman]{babel}
```

## Installing

Whenever inside a Linux-compatible environment, install texlive. Note that it is
recommended to compile texlive from source when using CentOs.

## BibTeX

The term BibTeX is a generic therm for bibliography packages and
compilers in LaTeX.

There are two sides: one for citing (inside LaTeX) and another for compiling.

- Compiling: `bibtex`, `biber`;
- Citing: `natbib`, `biblatex`

`natbib` package:

- It has been around for quite a long time, and although still
maintained, it is fair to say that it isn't being further developed;
- It is still widely used, and very reliable;
- It generates `.bst` files;
- It can be used with the `makebst` utility;
- It **requires `bibtex` for compiling**.

`biblatex` package:

- It is being actively developed in conjunction with the biber backend;
- It can work with `biber` and `bibtex` (use options when compiling);
- It generates `.bst` files;
- It has many citation styles, including `biblatex-abnt`.

## Compiling

The normal workflow to compile a document with bibliography is:

1. Compile LaTeX;
2. Compile bibliography;
3. Compile LaTeX; and
4. Compile LaTeX;

Bibliography compiler: read above.

LaTeX compiler may be `latex`, `pdflatex`, `xelatex`, `lualatex`, etc.
Or you can use `latexmk` which is much simpler and requires only one command:
`latexmk -pdf` to compile to PDF.
You can also compile it continuously with `latexmk -pvc -pdf` (each time a
modification is made, it will automatically recompile). Then, to clean up
use `latexmk -c` to remove all temporary files, except the output, or
`latexmk -C` to remove all temporary files, including the output.

Note: `latexmk` may be set with a custom `latexmkrc`

Example:

Suppose I have two files: `main.tex` (uses `natbib`) and `bibliography.bib`.

To compile with `pdflatex`:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If I want to compile with `latexmk`:

```sh
latexmk main.tex
```

## Figure side-by-side

For two independent side-by-side figures, you can use two minipages inside a figure environment; for two subfigures, I would recommend the subcaption package with its subfigure environment; here's an example showing both approaches:

```tex
\documentclass{article}
\usepackage[demo]{graphicx}
\usepackage{caption}
\usepackage{subcaption}

\begin{document}

	\begin{figure}
		\centering
		\begin{subfigure}{.5\textwidth}
			\centering
			\includegraphics[width=.4\linewidth]{image1}
			\caption{A subfigure}
			\label{fig:sub1}
		\end{subfigure}%
		\begin{subfigure}{.5\textwidth}
			\centering
			\includegraphics[width=.4\linewidth]{image1}
			\caption{A subfigure}
			\label{fig:sub2}
		\end{subfigure}
		\caption{A figure with two subfigures}
		\label{fig:test}
	\end{figure}

	\begin{figure}
		\centering
		\begin{minipage}{.5\textwidth}
			\centering
			\includegraphics[width=.4\linewidth]{image1}
			\captionof{figure}{A figure}
			\label{fig:test1}
		\end{minipage}%
		\begin{minipage}{.5\textwidth}
			\centering
			\includegraphics[width=.4\linewidth]{image1}
			\captionof{figure}{Another figure}
			\label{fig:test2}
		\end{minipage}
	\end{figure}

\end{document}
```

## Code listing

Default environment is `verbatim`:

```tex
\begin{verbatim}
Text enclosed inside \texttt{verbatim} environment
\end{verbatim}
There is also the \verb|C:\Windows\system32| command
```

The `verb` command automatically cancels all underscore. Alternatively, you can
use `\texttt{}` - which does not cancels.

There is also the `listings` package, which is a more advanced form of `verbatim`.

```tex
\begin{lstlisting}[language=Python]
import numpy as np
\end{lstlisting}
```

Or:

```tex
\lstset{
	language=Python,
	basicstyle=\tiny,
	tabsize=4
}

\begin{lstlisting}
import numpy as np
\end{lstlisting}
```

Every indexation counts, even within LaTeX! Don't be surprised if your code
stays in the middle.

### code listing in beamer

For using `verbatim` and `listings` with beamer, one has to input the `fragile`
option to the frame:

```tex
\begin{frame}[fragile]{Frame Title}{Frame Subtitle}
	\begin{lstlisting}
import numpy as np
	\end{lstlisting}
\end{frame}
```

# font size

- [Font sizes, families and
styles](https://www.overleaf.com/learn/latex/Font_sizes,_families,_and_styles)

- `\tiny`
- `\scriptsize`
- `\footnotesize`
- `\small`
- `\normalsize`
- `\large`
- `\Large`
- `\LARGE`
- `\huge`
- `\Huge`

Example:

```tex
This text it {\huge huge}. Back to normal!
```

Additionally:

style|command|switch command
---|---|---
serif (roman)|`\textrm{}`|`\rmfamily`
sans serif|`\textsf{}`|`\sffamily`
typewriter (monospace)|`\texttt{}`|`\ttfamily`
medium|`\textmd{}`|`\mdseries`
bold|`\textbf{}`|`\bfseries`
upright|`\textup{}`|`\upshape`
italic|`\textit{}`|`\itshape`
slanted|`\textsl{}`|`\slshape`
small caps|`\textsc{}`|`\scshape`

# text alignment

- [Text alignment](https://www.overleaf.com/learn/latex/Text_alignment)

Alignment|Environment|Switch command|ragged2e environment|ragged2e switch command
-|-|-|-|-
Left|flushleft|`\raggedright`|FlushLeft|`\RaggedRight`
Right|flushright|`\raggedleft`|FlushRight|`\RaggedLeft`
Centre|center|`\centering`|Center|`\Centering`
Fully justified|||justify|`\justify`

Example:

```tex
\begin{flushleft}
	Hello, here is some text without a meaning. \\
\end{flushleft}
{\flushleft Here is anothe text without any meaning}
```

Note that you have to enter the new line character, even at the end of the text
is shorter than a whole line.

## text color

### Standard colors

```tex
\usepackage{xcolor}
{\color{blue} my text}
```

### predefined colors names

Parameters are:

- `dvipsnames`: colors from the dvips driver, [available
  colors](https://www.overleaf.com/learn/latex/Using_colours_in_LaTeX#Reference_guide)
- `svgnames`
- `x11names`
- `xdvi`
- `dvipdf`
- `pdftex`
- `dvipsone`
- `dviwin`
- `truetex`
- `xtex`

```tex
\usepackage[dvipsnames]{xcolor}
{\color{RubineRed} my text}
\textcolor{RubineRed}{my text again}
\colorbox{BruntOrange{this text}
```

### custom colors

- [LaTeX Color](https://latexcolor.com/)

```tex
\usepackage{xcolor}
\definecolor{mypink1}{rgb}{0.858, 0.188, 0.478}
\definecolor{mypink2}{RGB}{219, 48, 122}
\definecolor{mypink3}{cmyk}{0, 0.7808, 0.4429, 0.1412}
\definecolor{mygray}{gray}{0.6}
\definecolor{Mycolor2}{HTML}{00F9DE}
\colorlet{Mycolor1}{green!10!orange!90!} % color mixing
\textcolor{mypink1}{Pink with rgb}
```

### define foreground and background colors

```tex
\pagecolor{black}
\color{white}
```

## page number

simple, no page number `\pagenumbering{gobble}` or `\pagestyle{empty}`.

## icons - symbols

[Font Awesome: icons](https://fontawesome.com/icons)

```tex
\usepackage{fontawesome}
\faicon{git}
```

## creating a line

```tex
\rule[depth]{width}{height}
\rule{1cm}{.15mm}
```

## text margin adjustments

This can be done with `minipage` or a number of packages.

```tex
\begin{minipage}{minipage width}
\begin{minipage}{.8\textwidth} % example
```


`minipage` will not break page: if it is too long for a page, it will skip the
entire page and go to the next. In this case it is recommended to use the
following:

```tex
\usepackage{changepage}

\begin{adjustwidth}{how much to take from left side}{how much from the right}
\begin{adjustwidth}{}{.1\textwidth} % example
```
