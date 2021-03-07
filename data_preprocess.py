from collections import Counter
import argparse
import os


def arg_parser():
    """ takes user input """

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='data/ontonetes-4.0',
                        help='use this option to provide a path to data dir (default=data/ontonetes-4.0).')
    parser.add_argument("--out_dir", default='data/',
                        help="use this option to provide the output dir (default=data/)")

    args = parser.parse_args()
    return args


def create_tsv_and_fetch_info(args):
    """
    generates tsv file from conll files and collects information from the data while iterating over each sentence
    """

    sequence_len_list = []  #contains length of every sequence
    num_words_per_tag = Counter()
    data_path = os.path.join(os.getcwd(), args.data_dir)
    output_file = os.path.join(os.getcwd(), args.out_dir, 'ontonetes-4.0.tsv')

    with open(output_file, 'w+') as of:

        for file in os.listdir(data_path):
            if not file.endswith(".gold_conll"): # sanity check for conll files
                continue

            with open(os.path.join(data_path, file), 'r') as f:
                count = 0

                for sent in f:
                    if sent.startswith('#'):  # ignore lines starting with #
                        continue

                    if sent == '\n':  # marks the end of sequence
                        sequence_len_list.append(count)
                        count = 0  # reset the count
                        of.write('*\n')
                        continue
                    
                    # 2nd token is word index
                    # 3rd token is word
                    # 4th token is POS tag
                    tokens = sent.split()
                    num_words_per_tag[tokens[4]] += 1
                    of.write(f'{tokens[2]}\t{tokens[3]}\t{tokens[4]}\n')
                    count += 1  # update count of the sequence

    return num_words_per_tag, sorted(sequence_len_list)


def write_info_to_file(args, num_words_per_tag, sequence_len_list):
    total_words = sum(num_words_per_tag.values())
    output_file = os.path.join(os.getcwd(), args.out_dir, 'ontonetes-4.0.info')

    with open(output_file, 'w+') as f:
        f.write(f'Maximum sequence length: {sequence_len_list[-1]}\n')
        f.write(f'Minimum sequence length: {sequence_len_list[0]}\n')
        f.write(f'Mean sequence length: {sum(sequence_len_list) / len(sequence_len_list)}\n')
        f.write(f'Number of sequences: {len(sequence_len_list)}\n')
        f.write(f'\nTags: \n')

        for tag, count in num_words_per_tag.items():
            f.write(f'{tag}\t{(count / total_words) * 100}%\n')


if __name__ == '__main__':
    args = arg_parser()
    
    num_words_per_tag, sequence_len_list = create_tsv_and_fetch_info(args)
    write_info_to_file(args, num_words_per_tag, sequence_len_list)
