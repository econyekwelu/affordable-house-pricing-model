import csv

if __name__ == '__main__':
    with open('data/train.csv', 'r', newline='') as file:
        csv_reader = csv.reader(file)
        # Skip the header row if present
        next(csv_reader, None)
        row_count = 0
        column_count = 0
        for row in csv_reader:
            row_count += 1
            column_count = max(column_count, len(row))
        print(f"Row count: {row_count}")
        print(f"Column count: {column_count}")
