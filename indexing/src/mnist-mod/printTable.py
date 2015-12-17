import locale

def _formatNum(num):
    """Format a number according to given places.
    Adds commas, etc. Will truncate floats into ints!"""

    try:
        if isinstance(num, int):
            return locale.format("%.*f", (0, num), True)
        else:
            return locale.format("%.*f", (3, num), True)

    except (ValueError, TypeError):
        return str(num)

def _getMaxWidth(table, index):
    """Get the maximum width of the given column index"""
    return max([len(_formatNum(row[index])) for row in table])

def printTableDict(out, dicts, caption=None):
    '''
    Prints out a table of data, padded for alignment
    Each dictionary in the dicts list must have the same set of keys
    '''
    table = [dicts[0].keys()]
    for d in dicts:
        table.append([d[k] for k in table[0]])
    printTable(out, table, caption)

def printTable(out, table, caption=None):
    """Prints out a table of data, padded for alignment
    @param out: Output stream (file-like object)
    @param table: The table to print. A list of lists.
    Each row must have the same number of columns. """
    columnWidths = []

    # Calculate the column widths
    for columnIndex in range(len(table[0])):
        columnWidths.append(_getMaxWidth(table, columnIndex))

    # Print the table caption
    if caption is not None:
        tableWidth = sum(columnWidths) + 3*len(columnWidths) - 2
        print >> out, caption.center(tableWidth, '_')

    # Print the table
    for row in table:
        # left col
        if isinstance(row[columnIndex], basestring):
            print >> out, row[0].center(columnWidths[0]) + " ",
        else:
            print >> out, _formatNum(row[0]).center(columnWidths[0]) + ' ',
        # rest of the cols
        for columnIndex in range(1, len(row)):
            if isinstance(row[columnIndex], basestring):
                columnText = "  " + row[columnIndex].center(columnWidths[columnIndex])
            else:
                columnText = "  " + _formatNum(row[columnIndex]).center(columnWidths[columnIndex])
            print >> out, columnText,
        print >> out
