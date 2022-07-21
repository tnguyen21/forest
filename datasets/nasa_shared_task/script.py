import csv

if __name__ == "__main__":
    dict_txt_path = "HEXTRATO_dictionary.txt"
    csv_output_path = "HEXTRATO_dictionary.csv"


    with open(dict_txt_path, "r") as txt_file:
        with open(csv_output_path, "w") as csv_file:
            output_writer = csv.writer(csv_file, delimiter=",")

            for line in txt_file:
                if line == "\n":
                    continue
                stripped_line = line.strip()
                # non-ascii characters giving headache
                # TODO is there another way to avoid getting encoding errors
                # with any arbitrary dictionary? otherwise the system will
                # be limited
                ascii_only_line = stripped_line.encode('ascii', 'ignore').decode('ascii')
                csv_row = [ascii_only_line, ascii_only_line]
                output_writer.writerow(csv_row)

