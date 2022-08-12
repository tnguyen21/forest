import csv

if __name__ == "__main__":
    dict_txt_path = "HEXTRATO_dictionary.txt"
    csv_output_path = "HEXTRATO_dictionary.csv"

    # ratio of training/test data
    train = 0.6
    tune = 0.2
    # test = 0.2

    expression_list = []

    with open(dict_txt_path, "r") as txt_file:
        for line in txt_file:
            if line == "\n":
                continue
            stripped_line = line.strip()
            # non-ascii characters giving headache
            # TODO is there another way to avoid getting encoding errors
            # with any arbitrary dictionary? otherwise the system will
            # be limited
            ascii_only_line = stripped_line.encode('ascii', 'ignore').decode('ascii')
            # csv_row = [ascii_only_line, ascii_only_line]
            expression_list.append(ascii_only_line)
    
    train_size = int(len(expression_list) * train)
    tune_size = int(len(expression_list) * tune)
    test_size = int(len(expression_list) - train_size - tune_size)

    with open("training.csv", "w") as csv_file:
        output_writer = csv.writer(csv_file, delimiter=",")

        for i in range(train_size):
            csv_row = [expression_list[i], expression_list[i]]
            output_writer.writerow(csv_row)
    
    with open("tuning.csv", "w") as csv_file:
        output_writer = csv.writer(csv_file, delimiter=",")

        for i in range(train_size, train_size + tune_size):
            csv_row = [expression_list[i], expression_list[i]]
            output_writer.writerow(csv_row)
    
    with open("test.csv", "w") as csv_file:
        output_writer = csv.writer(csv_file, delimiter=",")

        for i in range(train_size + tune_size, train_size + tune_size + test_size):
            csv_row = [expression_list[i], expression_list[i]]
            output_writer.writerow(csv_row)


