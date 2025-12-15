/**
 * Output formatters for JadeVectorDB JavaScript CLI
 *
 * Provides formatting functions for different output formats
 * (JSON, YAML, table, CSV) to improve CLI usability and integration.
 */

const yaml = require('js-yaml');
const Table = require('cli-table3');

/**
 * Format data as JSON
 */
function formatJson(data, indent = 2) {
  return JSON.stringify(data, null, indent);
}

/**
 * Format data as YAML
 */
function formatYaml(data) {
  try {
    return yaml.dump(data, {
      sortKeys: false,
      lineWidth: -1
    });
  } catch (error) {
    console.error('Warning: Error formatting YAML, falling back to JSON');
    return formatJson(data);
  }
}

/**
 * Format data as a table
 */
function formatTable(data, headers = null) {
  try {
    // Handle different data types
    if (Array.isArray(data)) {
      if (data.length === 0) {
        return 'No data to display';
      }

      // Array of objects - show as table
      if (typeof data[0] === 'object' && data[0] !== null) {
        const actualHeaders = headers || Object.keys(data[0]);
        const table = new Table({
          head: actualHeaders,
          style: { head: ['cyan'] }
        });

        data.forEach(item => {
          const row = actualHeaders.map(h => {
            const value = item[h];
            if (value === null || value === undefined) return '';
            if (typeof value === 'object') return JSON.stringify(value);
            return String(value);
          });
          table.push(row);
        });

        return table.toString();
      }

      // Array of primitives
      const table = new Table({
        head: ['Value'],
        style: { head: ['cyan'] }
      });
      data.forEach(item => table.push([String(item)]));
      return table.toString();

    } else if (typeof data === 'object' && data !== null) {
      // Single object - show as key-value pairs
      const table = new Table({
        head: ['Key', 'Value'],
        style: { head: ['cyan'] }
      });

      Object.entries(data).forEach(([key, value]) => {
        const displayValue = typeof value === 'object' ?
          JSON.stringify(value) : String(value);
        table.push([key, displayValue]);
      });

      return table.toString();
    }

    // For other types, convert to string
    return String(data);

  } catch (error) {
    console.error('Warning: Error formatting table, falling back to JSON');
    return formatJson(data);
  }
}

/**
 * Format data as CSV
 */
function formatCsv(data, headers = null) {
  try {
    const lines = [];

    // Handle different data types
    if (Array.isArray(data)) {
      if (data.length === 0) {
        return '';
      }

      // Array of objects - format as CSV table
      if (typeof data[0] === 'object' && data[0] !== null) {
        const actualHeaders = headers || Object.keys(data[0]);

        // Add header row
        lines.push(actualHeaders.map(h => escapeCSV(h)).join(','));

        // Add data rows
        data.forEach(item => {
          const row = actualHeaders.map(h => {
            const value = item[h];
            if (value === null || value === undefined) return '';
            if (typeof value === 'object') return escapeCSV(JSON.stringify(value));
            return escapeCSV(String(value));
          });
          lines.push(row.join(','));
        });
      } else {
        // Array of primitives
        lines.push('Value');
        data.forEach(item => {
          lines.push(escapeCSV(String(item)));
        });
      }

    } else if (typeof data === 'object' && data !== null) {
      // Single object - format as key-value pairs
      lines.push('Key,Value');
      Object.entries(data).forEach(([key, value]) => {
        const displayValue = typeof value === 'object' ?
          JSON.stringify(value) : String(value);
        lines.push(`${escapeCSV(key)},${escapeCSV(displayValue)}`);
      });
    } else {
      // For other types, convert to string
      return String(data);
    }

    return lines.join('\n');

  } catch (error) {
    console.error('Warning: Error formatting CSV, falling back to JSON');
    return formatJson(data);
  }
}

/**
 * Escape CSV values (handle quotes and commas)
 */
function escapeCSV(value) {
  const str = String(value);
  // If value contains comma, quote, or newline, wrap in quotes and escape quotes
  if (str.includes(',') || str.includes('"') || str.includes('\n')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/**
 * Format data according to specified output format
 */
function formatOutput(data, outputFormat = 'json', headers = null) {
  const format = outputFormat.toLowerCase();

  switch (format) {
    case 'json':
      return formatJson(data);
    case 'yaml':
      return formatYaml(data);
    case 'table':
      return formatTable(data, headers);
    case 'csv':
      return formatCsv(data, headers);
    default:
      throw new Error(`Unsupported output format: ${outputFormat}`);
  }
}

/**
 * Print formatted data to stdout
 */
function printFormatted(data, outputFormat = 'json', headers = null) {
  const formatted = formatOutput(data, outputFormat, headers);
  console.log(formatted);
}

module.exports = {
  formatJson,
  formatYaml,
  formatTable,
  formatCsv,
  formatOutput,
  printFormatted
};
