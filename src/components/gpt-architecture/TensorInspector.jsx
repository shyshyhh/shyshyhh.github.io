import { useId, useMemo, useRef, useState } from 'react';

const MAX_ROWS = 32;
const MAX_COLUMNS = 32;

const COLOR_HUES = Object.freeze({
  amber: 42,
  blue: 214,
  cyan: 188,
  emerald: 154,
  fuchsia: 296,
  green: 142,
  indigo: 238,
  lime: 88,
  orange: 28,
  pink: 330,
  purple: 278,
  red: 5,
  rose: 348,
  teal: 174,
  violet: 266,
  yellow: 54,
});

function normalizeHue(hue) {
  return ((hue % 360) + 360) % 360;
}

function hexToHue(value) {
  const match = value.match(/^#([\da-f]{3}|[\da-f]{6})$/i);
  if (!match) return null;

  const hex =
    match[1].length === 3
      ? [...match[1]].map((character) => character.repeat(2)).join('')
      : match[1];
  const [red, green, blue] = [0, 2, 4].map(
    (offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255
  );
  const maximum = Math.max(red, green, blue);
  const minimum = Math.min(red, green, blue);
  const range = maximum - minimum;

  if (range === 0) return 188;

  let hue;
  if (maximum === red) hue = ((green - blue) / range) % 6;
  else if (maximum === green) hue = (blue - red) / range + 2;
  else hue = (red - green) / range + 4;

  return normalizeHue(hue * 60);
}

function resolveHue(color) {
  if (typeof color === 'number' && Number.isFinite(color)) {
    return normalizeHue(color);
  }

  const normalized = String(color || 'cyan').trim().toLowerCase();
  if (Object.hasOwn(COLOR_HUES, normalized)) return COLOR_HUES[normalized];

  const numericHue = Number.parseFloat(normalized);
  if (Number.isFinite(numericHue)) return normalizeHue(numericHue);

  return hexToHue(normalized) ?? COLOR_HUES.cyan;
}

function normalizeMatrix(matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0) return [];

  const rows = Array.isArray(matrix[0]) ? matrix : [matrix];
  return rows.slice(0, MAX_ROWS).map((row) =>
    (Array.isArray(row) ? row : [row])
      .slice(0, MAX_COLUMNS)
      .map((value) => Number(value))
  );
}

function exactValue(value) {
  if (Object.is(value, -0)) return '-0';
  return String(value);
}

function compactValue(value) {
  if (Number.isNaN(value)) return 'NaN';
  if (value === Number.POSITIVE_INFINITY) return '+∞';
  if (value === Number.NEGATIVE_INFINITY) return '−∞';
  if (Object.is(value, -0) || value === 0) return '0';

  const magnitude = Math.abs(value);
  let formatted;

  if (magnitude >= 100 || magnitude < 0.01) {
    formatted = magnitude.toExponential(1);
  } else if (magnitude >= 10) {
    formatted = magnitude.toFixed(1);
  } else {
    formatted = magnitude.toFixed(2);
  }

  return `${value > 0 ? '+' : '−'}${formatted}`;
}

function labelAt(labels, index, fallback) {
  const label = labels?.[index];
  return label === undefined || label === null || label === ''
    ? fallback
    : String(label);
}

/**
 * Compact, inspectable rendering for the deliberately small tensors in the
 * GPT explorer. Vectors are presented as a one-row matrix.
 */
export default function TensorInspector({
  title = 'Tensor',
  subtitle,
  matrix,
  rowLabels,
  columnLabels,
  color = 'cyan',
  onCellSelect,
}) {
  const titleId = useId();
  const [selection, setSelection] = useState({ row: 0, column: 0 });
  const cellRefs = useRef(new Map());

  const values = useMemo(() => normalizeMatrix(matrix), [matrix]);
  const baseHue = useMemo(() => resolveHue(color), [color]);
  const rowCount = values.length;
  const columnCount = values.reduce(
    (maximum, row) => Math.max(maximum, row.length),
    0
  );
  const maximumMagnitude = values.reduce(
    (maximum, row) =>
      row.reduce(
        (rowMaximum, value) =>
          Number.isFinite(value)
            ? Math.max(rowMaximum, Math.abs(value))
            : rowMaximum,
        maximum
      ),
    0
  );

  const selectionExists =
    selection.row < rowCount &&
    selection.column < (values[selection.row]?.length ?? 0);
  const activeSelection = selectionExists
    ? selection
    : rowCount > 0 && values[0].length > 0
      ? { row: 0, column: 0 }
      : null;
  const activeValue = activeSelection
    ? values[activeSelection.row][activeSelection.column]
    : null;

  const selectCell = (row, column) => {
    const value = values[row][column];
    setSelection({ row, column });
    onCellSelect?.({
      row,
      column,
      value,
      rowLabel: labelAt(rowLabels, row, `Row ${row + 1}`),
      columnLabel: labelAt(columnLabels, column, `Column ${column + 1}`),
    });
  };

  const focusCell = (row, column) => {
    if (!values[row]?.length) return;
    const boundedColumn = Math.min(
      Math.max(column, 0),
      values[row].length - 1
    );
    selectCell(row, boundedColumn);
    window.requestAnimationFrame(() => {
      cellRefs.current.get(`${row}:${boundedColumn}`)?.focus();
    });
  };

  const handleCellKeyDown = (event, row, column) => {
    let nextRow = row;
    let nextColumn = column;

    if (event.key === 'ArrowLeft') {
      nextColumn = Math.max(0, column - 1);
    } else if (event.key === 'ArrowRight') {
      nextColumn = Math.min(values[row].length - 1, column + 1);
    } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      const direction = event.key === 'ArrowUp' ? -1 : 1;
      let candidate = row + direction;
      while (
        candidate >= 0 &&
        candidate < rowCount &&
        !values[candidate]?.length
      ) {
        candidate += direction;
      }
      if (candidate < 0 || candidate >= rowCount) return;
      nextRow = candidate;
      nextColumn = Math.min(column, values[candidate].length - 1);
    } else if (event.key === 'Home') {
      if (event.ctrlKey || event.metaKey) {
        nextRow = values.findIndex((candidate) => candidate.length > 0);
      }
      nextColumn = 0;
    } else if (event.key === 'End') {
      if (event.ctrlKey || event.metaKey) {
        nextRow = rowCount - 1;
        while (nextRow > 0 && !values[nextRow]?.length) nextRow -= 1;
      }
      nextColumn = values[nextRow].length - 1;
    } else {
      return;
    }

    event.preventDefault();
    focusCell(nextRow, nextColumn);
  };

  return (
    <section className="gptx-tensor" aria-labelledby={titleId}>
      <header className="gptx-tensor-header">
        <div className="gptx-tensor-heading">
          <h3 className="gptx-tensor-title" id={titleId}>
            {title}
          </h3>
          {subtitle ? (
            <p className="gptx-tensor-subtitle">{subtitle}</p>
          ) : null}
        </div>
        <span
          className="gptx-tensor-shape"
          aria-label={`Tensor shape: ${rowCount} by ${columnCount}`}
        >
          {rowCount} × {columnCount}
        </span>
      </header>

      {rowCount > 0 && columnCount > 0 ? (
        <div className="gptx-tensor-scroll">
          <table className="gptx-tensor-grid" aria-labelledby={titleId}>
            {columnLabels ? (
              <thead className="gptx-tensor-grid-head">
                <tr className="gptx-tensor-grid-row">
                  {rowLabels ? (
                    <th
                      className="gptx-tensor-grid-corner"
                      aria-hidden="true"
                    />
                  ) : null}
                  {Array.from({ length: columnCount }, (_, column) => (
                    <th
                      className="gptx-tensor-column-label"
                      scope="col"
                      key={`column-${column}`}
                    >
                      {labelAt(columnLabels, column, `c${column}`)}
                    </th>
                  ))}
                </tr>
              </thead>
            ) : null}
            <tbody className="gptx-tensor-grid-body">
              {values.map((row, rowIndex) => (
                <tr className="gptx-tensor-grid-row" key={`row-${rowIndex}`}>
                  {rowLabels ? (
                    <th className="gptx-tensor-row-label" scope="row">
                      {labelAt(rowLabels, rowIndex, `r${rowIndex}`)}
                    </th>
                  ) : null}
                  {row.map((value, columnIndex) => {
                    const isSelected =
                      activeSelection?.row === rowIndex &&
                      activeSelection?.column === columnIndex;
                    const intensity = Number.isFinite(value)
                      ? maximumMagnitude > 0
                        ? Math.abs(value) / maximumMagnitude
                        : 0
                      : 1;
                    const sign =
                      value > 0 ? 'positive' : value < 0 ? 'negative' : 'zero';
                    const hue =
                      value < 0 ? normalizeHue(baseHue + 160) : baseHue;
                    const rowLabel = labelAt(
                      rowLabels,
                      rowIndex,
                      `Row ${rowIndex + 1}`
                    );
                    const columnLabel = labelAt(
                      columnLabels,
                      columnIndex,
                      `Column ${columnIndex + 1}`
                    );
                    const accessibleLabel = `${rowLabel}, ${columnLabel}: ${exactValue(value)}`;

                    return (
                      <td
                        className="gptx-tensor-grid-cell"
                        key={`cell-${rowIndex}-${columnIndex}`}
                      >
                        <button
                          ref={(node) => {
                            const key = `${rowIndex}:${columnIndex}`;
                            if (node) cellRefs.current.set(key, node);
                            else cellRefs.current.delete(key);
                          }}
                          type="button"
                          className={`gptx-tensor-cell gptx-tensor-cell--${sign}${
                            isSelected
                              ? ' gptx-tensor-cell--selected'
                              : ''
                          }`}
                          style={{
                            '--gptx-cell-hue': hue,
                            '--gptx-cell-intensity': intensity.toFixed(4),
                          }}
                          aria-label={accessibleLabel}
                          aria-pressed={isSelected}
                          tabIndex={isSelected ? 0 : -1}
                          title={accessibleLabel}
                          onClick={() => selectCell(rowIndex, columnIndex)}
                          onKeyDown={(event) =>
                            handleCellKeyDown(
                              event,
                              rowIndex,
                              columnIndex
                            )
                          }
                        >
                          <span className="gptx-tensor-cell-value">
                            {compactValue(value)}
                          </span>
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="gptx-tensor-empty">No tensor values to inspect.</p>
      )}

      <div
        className="gptx-tensor-readout"
        aria-live="polite"
        aria-atomic="true"
      >
        <span className="gptx-tensor-readout-label">Selected value</span>
        {activeSelection ? (
          <>
            <span className="gptx-tensor-readout-coordinate">
              {labelAt(
                rowLabels,
                activeSelection.row,
                `r${activeSelection.row}`
              )}
              {' · '}
              {labelAt(
                columnLabels,
                activeSelection.column,
                `c${activeSelection.column}`
              )}
            </span>
            <output className="gptx-tensor-readout-value">
              {exactValue(activeValue)}
            </output>
          </>
        ) : (
          <output className="gptx-tensor-readout-value">—</output>
        )}
      </div>
    </section>
  );
}
