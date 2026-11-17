// Build the paper's .docx from the block specification written by build.py.
//
//   node docx_builder.js build/spec.json build/paper.docx
//
// The specification holds only resolved text (every number already filled in from the results files), so this
// script does layout only: A4, Times New Roman, centred title block, justified abstract, numbered headings,
// tables in the three-rule style of journals (heavy rule above and below the table, a light rule under the header,
// no vertical lines) with the caption above, figures with the caption below.
"use strict";
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat, Packer, PageNumber,
  Paragraph, ShadingType, Table, TableCell, TableRow, TextRun, VerticalAlign, WidthType,
} = require("docx");

const FONT = "Times New Roman";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1300;          // A4, about 2.3 cm margins (DXA)
const TEXT_W = PAGE_W - 2 * MARGIN;
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const HEAVY = { style: BorderStyle.SINGLE, size: 12, color: "000000" };   // 1.5 pt
const LIGHT = { style: BorderStyle.SINGLE, size: 4, color: "000000" };    // 0.5 pt

function runs(spec, base = {}) {
  return (spec || []).map((r) => new TextRun({
    text: r.t, bold: r.b || base.bold, italics: r.i || base.italics, superScript: r.sup || false,
    font: FONT, size: base.size || 22,
  }));
}

// "< 0.001", "55.0 ± 1.0" and "[8.3, 25.6]" each stay on one line
const keepTogether = (spec) => (spec || []).map((r) => ({ ...r, t: r.t
  .replace(/< (?=\d)/g, "<\u00a0").replace(/ ± /g, "\u00a0±\u00a0").replace(/\[([^\]]*?), ([^\]]*?)\]/g, "[$1,\u00a0$2]") }));

function cellParagraphs(cell, opts) {
  return [new Paragraph({
    alignment: opts.align, spacing: { before: 30, after: 30 }, keepNext: opts.keepNext, keepLines: true,
    children: runs(keepTogether(cell), { size: 18, bold: opts.bold }),
  })];
}

function table(block) {
  const widths = block.widths;
  const total = widths.reduce((a, b) => a + b, 0);
  if (total !== TEXT_W) throw new Error(`table "${block.name}": widths sum to ${total}, expected ${TEXT_W}`);
  const last = block.rows.length - 1;
  // first column and prose columns (long cells) left-aligned, short numeric columns centred
  const textLen = (cell) => (cell || []).map((r) => r.t).join("").length;
  const prose = widths.map((_, j) => j === 0 ||
    block.rows.reduce((a, r) => a + textLen(r[j]), 0) / Math.max(1, block.rows.length) > 28);
  const row = (cells, header, i) => new TableRow({
    tableHeader: header, cantSplit: true,
    children: cells.map((c, j) => new TableCell({
      borders: { left: NONE, right: NONE, top: header ? HEAVY : NONE,
                 bottom: header ? LIGHT : (i === last ? HEAVY : NONE) },
      width: { size: widths[j], type: WidthType.DXA },
      verticalAlign: header ? VerticalAlign.BOTTOM : VerticalAlign.TOP,
      margins: { top: header ? 60 : 25, bottom: header ? 60 : 25, left: 90, right: 90 },
      // keep a table on one page: every row but the last keeps with the next
      children: cellParagraphs(c, { bold: header, keepNext: header || i < last,
        align: prose[j] ? AlignmentType.LEFT : AlignmentType.CENTER }),
    })),
  });
  return [
    new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 }, children: runs(block.caption, { italics: true, size: 19 }) }),
    new Table({
      width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
      borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
      rows: [row(block.header, true, -1), ...block.rows.map((r, i) => row(r, false, i))],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

function figure(block, root) {
  const file = path.resolve(root, block.path);
  const data = fs.readFileSync(file);
  const w = Math.round(block.width_px), h = Math.round(block.width_px * block.aspect);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 60 },
      children: [new ImageRun({ type: "png", data, transformation: { width: w, height: h },
        altText: { title: block.alt, description: block.alt, name: path.basename(file) } })] }),
    new Paragraph({ spacing: { after: 160 }, children: runs(block.caption, { italics: true, size: 19 }) }),
  ];
}

function build(spec, root) {
  const out = [];
  const m = spec.meta;
  out.push(new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: m.title, bold: true, font: FONT, size: 32 })] }));
  out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: m.author, font: FONT, size: 22 })] }));
  out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: m.affil, italics: true, font: FONT, size: 19 })] }));
  let prev = null, listInstance = 0;
  for (const b of spec.blocks) {
    switch (b.type) {
      case "abstract":
        out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
          children: [new TextRun({ text: "Abstract", bold: true, font: FONT, size: 21 })] }));
        out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 567, right: 567 },
          spacing: { after: 240, line: 264 }, children: runs(b.runs, { size: 19 }) }));
        break;
      case "h1":
        out.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, spacing: { before: 280, after: 120 },
          children: [new TextRun({ text: b.text, bold: true, font: FONT, size: 26 })] }));
        break;
      case "h2":
        out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 200, after: 80 },
          children: [new TextRun({ text: b.text, bold: true, font: FONT, size: 22 })] }));
        break;
      case "p":
        out.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 264 },
          children: runs(b.runs) }));
        break;
      case "bullet":
        out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
          spacing: { after: 60, line: 264 }, children: runs(b.runs) }));
        break;
      case "number":
        if (prev !== "number") listInstance += 1;          // each numbered list starts again at 1
        out.push(new Paragraph({ numbering: { reference: "steps", level: 0, instance: listInstance }, alignment: AlignmentType.JUSTIFIED,
          spacing: { after: 60, line: 264 }, children: runs(b.runs) }));
        break;
      case "code":
        out.push(new Paragraph({ spacing: { before: 40, after: 40, line: 240 }, indent: { left: 400 },
          shading: { type: ShadingType.CLEAR, color: "auto", fill: "F2F4F7" },
          children: [new TextRun({ text: b.text, font: "Courier New", size: 17 })] }));
        break;
      case "ref":
        out.push(new Paragraph({ spacing: { after: 60 }, indent: { left: 400, hanging: 400 },
          children: runs(b.runs, { size: 19 }) }));
        break;
      case "table": out.push(...table(b)); break;
      case "figure": out.push(...figure(b, root)); break;
      default: throw new Error(`unknown block type ${b.type}`);
    }
    prev = b.type;
  }
  return new Document({
    creator: m.author, title: m.title,
    styles: {
      default: { document: { run: { font: FONT, size: 22 } } },
      paragraphStyles: [
        { id: "Title", name: "Title", basedOn: "Normal", run: { font: FONT, size: 32, bold: true },
          paragraph: { spacing: { before: 0, after: 120 }, keepNext: true } },
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 22, bold: true }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 1 } },
      ],
    },
    numbering: { config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
      { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 300 } } } }] },
    ] },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
      children: out,
    }],
  });
}

if (require.main === module) {
  const [specPath, outPath] = process.argv.slice(2);
  if (!specPath || !outPath) {
    console.error("usage: node docx_builder.js spec.json out.docx");
    process.exit(2);
  }
  const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
  Packer.toBuffer(build(spec, path.dirname(path.resolve(specPath, "..")))).then((buf) => {
    fs.writeFileSync(outPath, buf);
    console.log(outPath);
  });
}

module.exports = { build, TEXT_W };
