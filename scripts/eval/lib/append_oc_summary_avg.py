#!/usr/bin/env python3
"""Append an overall_average row to an OpenCompass summary_*.txt file."""

import csv
import io
import sys

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def main():
    summary_file = sys.argv[1]
    with open(summary_file, "r", encoding="utf-8") as fin:
        text = fin.read()
    if "overall_average" in text:
        return

    csv_marker = "csv format\n"
    divider_marker = "$" * 124
    table_marker = "tabulate format\n"
    csv_start = text.find(csv_marker)
    if csv_start < 0:
        return

    csv_after_marker = text[csv_start + len(csv_marker) :]
    csv_caret_end = csv_after_marker.find("\n")
    if csv_caret_end < 0:
        return

    csv_preamble = text[: csv_start + len(csv_marker) + csv_caret_end + 1]
    csv_text = csv_after_marker[csv_caret_end + 1 :].lstrip("\n")
    csv_lines = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        if "," not in line:
            if csv_lines:
                break
            continue
        csv_lines.append(line)
    if not csv_lines:
        return

    reader = csv.reader(io.StringIO("\n".join(csv_lines)))
    rows = list(reader)
    if len(rows) < 2:
        return

    header = rows[0]
    data_rows = rows[1:]
    value_start = 4
    if len(header) <= value_start:
        return

    column_sums = [0.0] * (len(header) - value_start)
    column_counts = [0] * (len(header) - value_start)
    for row in data_rows:
        if not row or row[0] == "overall_average":
            continue
        for i in range(value_start, min(len(row), len(header))):
            cell = row[i].strip()
            if not cell:
                continue
            try:
                value = float(cell)
            except ValueError:
                continue
            column_sums[i - value_start] += value
            column_counts[i - value_start] += 1

    metric = next((row[2] for row in data_rows if len(row) > 2 and row[0] != "overall_average"), "average")
    mode = next((row[3] for row in data_rows if len(row) > 3 and row[0] != "overall_average"), "average")
    overall_row = ["overall_average", "-", metric, mode]
    for total, count in zip(column_sums, column_counts):
        overall_row.append(f"{(total / count):.2f}" if count else "")

    csv_rows = rows + [overall_row]
    csv_output = io.StringIO()
    writer = csv.writer(csv_output, lineterminator="\n")
    writer.writerows(csv_rows)
    csv_block = csv_output.getvalue().rstrip("\n")

    table_start = text.find(table_marker)
    if table_start >= 0:
        table_after_marker = text[table_start + len(table_marker) :]
        table_caret_end = table_after_marker.find("\n")
        if table_caret_end >= 0:
            table_preamble = text[: table_start + len(table_marker) + table_caret_end + 1]
            divider_start = text.find(divider_marker)
            if divider_start >= 0:
                if tabulate is not None:
                    table_block = tabulate(
                        data_rows + [overall_row],
                        headers=header,
                        tablefmt="simple",
                        stralign="left",
                        numalign="right",
                    )
                else:
                    table_rows = [header] + data_rows + [overall_row]
                    widths = [0] * len(header)
                    for row in table_rows:
                        for i, cell in enumerate(row):
                            widths[i] = max(widths[i], len(str(cell)))

                    def format_table_row(row):
                        parts = []
                        for i, cell in enumerate(row):
                            cell = str(cell)
                            if i >= value_start:
                                parts.append(cell.rjust(widths[i]))
                            else:
                                parts.append(cell.ljust(widths[i]))
                        return "  ".join(parts)

                    separator = "  ".join("-" * width for width in widths)
                    table_block = "\n".join(
                        [format_table_row(header), separator]
                        + [format_table_row(row) for row in data_rows + [overall_row]]
                    )
                text = table_preamble + table_block + "\n" + text[divider_start:]

    csv_start = text.find(csv_marker)
    csv_after_marker = text[csv_start + len(csv_marker) :]
    csv_caret_end = csv_after_marker.find("\n")
    csv_preamble = text[: csv_start + len(csv_marker) + csv_caret_end + 1]
    text = csv_preamble + csv_block + "\n"

    with open(summary_file, "w", encoding="utf-8") as fout:
        fout.write(text)


if __name__ == "__main__":
    main()
