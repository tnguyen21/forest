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
                csv_row = [stripped_line, stripped_line]
                output_writer.writerow(csv_row)

