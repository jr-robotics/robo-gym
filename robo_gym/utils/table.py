import csv


class RowIterator(object):
    def __init__(self, table, starting_row_index=-1):
        self.table = table
        self.current_row_index = starting_row_index

    def __next__(self):
        self.current_row_index += 1
        if not self.has_current_row():
            raise StopIteration
        result = self.get_current_row()
        return result

    def get_current_row(self):
        return self.table.get_row(self.current_row_index)

    def has_current_row(self):
        return self.table and self.current_row_index < self.table.count_rows()

    def has_next_row(self):
        return self.table and self.current_row_index + 1 < self.table.count_rows()

    def get_columns(self):
        return self.table.get_columns()


class Table(object):

    def __init__(self, rows=[], columns=[]):
        self.rows = rows  # list of dicts
        self.columns = columns
        if not self.columns:
            if self.rows:
                self.columns = self.rows[0].keys

    def __iter__(self):
        return RowIterator(self)

    def get_row(self, row_index):
        return self.rows[row_index]

    def add_row(self, row):
        self.rows.append(row)
        if not self.columns:
            self.columns = row.keys()

    def count_rows(self):
        return len(self.rows)

    def get_columns(self):
        return self.columns

    def clear_rows(self):
        self.rows = []


def write_csv(table, file_path, delimiter=",", value_formatters=None):
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, table.columns, delimiter=delimiter)
        writer.writeheader()
        for row in table:
            processed_row = row
            for column in table.columns:
                value = row[column]
                if value_formatters is None:
                    value = str(value)
                elif callable(value_formatters):
                    value = value_formatters(value)
                elif column in value_formatters and callable(value_formatters[column]):
                    value = value_formatters[column](value)
                processed_row[column] = value
            writer.writerow(processed_row)


def read_csv(file_path, delimiter=",", value_parsers=None):
    with open(file_path, "r") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        table = Table([], reader.fieldnames)
        for row in reader:
            processed_row = row
            for column in reader.fieldnames:
                value = row[column]
                if value_parsers is None:
                    value = float(value)
                elif callable(value_parsers):
                    value = value_parsers(value)
                elif column in value_parsers and callable(value_parsers[column]):
                    value = value_parsers[column](value)
                processed_row[column] = value
            table.add_row(processed_row)
        return table
