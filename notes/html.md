# HTML

Open-close tags:

- `<html>` encloses the HTML content: the head and the body
- `<head>` encloses the title and UTF-8 coding
- `<title>` the title, shown as the title of the browser's tag
- `<body>` document's body, with headers, paragraphs, etc
- `<h1>`, `<h2>`, `<h3>`, etc: the headers
- `<p>` paragraph
- `<pre>` present text just as original in HTML, useful for code
- `<div>` (division) group, useful for applying attributes to elements as one
- `<span>` inline group, similar to `<div>`
- `<i>` italic
- `<u>` underline
- `<b>` bold
- `<strike>` strikethrough
- `<tt>` monospace
- `<sup>` superscript
- `<sub>` subscript
- `<ins>` inserted text (may appear as underlined)
- `<del>` deleted text (may appear as strikethrough)
- `<em>` emphasized text (may appear as italic)
- `<strong>` strong text (may appear as bold)
- `<mark>` marked text, may appear with yellow background
- `<big>` one font size larger
- `<small>` one font size smaller
- `<abbr title="Full text">` abbreviation of text with full description
- `<acronym>` acronym, may not change appearance at all
- `<bdo>` invert characters horizontally
- `<dfn>` definition term (may appear as italic)
- `<blockquote>` quote block, just like in a thesis
- `<q>` inline quote, may appear as double quotes
- `<cite>` citation text (may appear as italic)
- `<code>` code (may appear as monospace)
- `<kbd>` keyboard text (may appear as monospace)
- `<var>` variable, used inside `<code>` (may appear as monospace italic)
- `<samp>` program output (may appear as monospace)
- `<address>` address (may appear as italic)
- `<ol>` ordered list
- `<ul>` unordered list
- `<li>` list item for both ol and ul

Void elements (single-use) tags and elements:

- `<!DOCTYPE html>` defines the document type and version
- `<br />` like break, used inside paragraph
- `<hr />` horizontal line
- `<a href="/link/to/page>` hyperlink
- `&nbsp` nonbreaking space
- `<meta>` see below
- `<img src="./my/image.svg">` add an image, see respective section

## attribute

Tags also have attributes, which can be set. Every attribute consists of

- name: propriety to be set
- value: value of the propriety

```html
<p align="center">Center aligned</p>
```

Core attributes are:

- id: unique identification
- title: similar to id, but shown as tip when mouse hovers over element
- class: associate element with CSS
- style: style, i.e. font-family, color...

Internationalization attributes (`<!DOCTYPE>` tag) are:

- dir: direction: left to right (`ltr`) or right to left (`rtl`)
- lang: document language, according to ISO 639
- align: horizontal alignment
- valign: vertical alignment
- bgcolor: background color behind element
- background: background image behind element
- width: width of tables, images or table cells
- height: height of tables, images or table cells
- title: Pop-up title of element

## tag vs. element

An element is defined by a starting tag. If the element contains other content,
it ends with a closing tag.

## `<meta/>`

Provides additional metadata information. Does not change appearance of page.

- `<meta charset="utf-8"/>`

## image

```html
<img src="./my/image.gif" alt="An alternative in case the image doen't load" title="A textual hint about the image."
```
