from collections import Counter
import argparse

def arg_parser():
    """ takes user input """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default='data/sample.conll',
                        help='use this option to provide a file (in CONLL format)')
    parser.add_argument("--output_dir", default='data/',
                        help="use this option to provide the output directory path")

    args = parser.parse_args()
    return args


def create_tsv_and_fetch_info(input_file, out_dir):
    """
    generates sample.tsv file from sample.conll and collects information from the data while iterating over each sentence
    """
    sequence_len_list = []  #contains length of every sequence
    num_words_per_tag = Counter()

    with open(out_dir+'sample.tsv', 'w+') as of:
        with open(input_file, 'r') as f:
            sents = f.readlines()
            count = 0

            for i in range(len(sents)):

                if sents[i].startswith('#'):  # ignore lines starting with #
                    continue

                if sents[i] == '\n':  # marks the end of sequence
                    sequence_len_list.append(count)
                    count = 0  # reset the count
                    of.write('*\n')
                    continue

                tokens = sents[i].split()
                num_words_per_tag[tokens[4]] += 1
                of.write(f'{tokens[2]}\t{tokens[3]}\t{tokens[4]}\n')
                count += 1  # update count of the sequence

    return num_words_per_tag, sorted(sequence_len_list)


def write_info_to_file(out_dir, num_words_per_tag, sequence_len_list):
    total_words = sum(num_words_per_tag.values())

    with open(out_dir+'sample.info','w+') as f:
        f.write(f'Maximum sequence length: {sequence_len_list[-1]}\n')
        f.write(f'Minimum sequence length: {sequence_len_list[0]}\n')
        f.write(f'Mean sequence length: {sum(sequence_len_list) / len(sequence_len_list)}\n')
        f.write(f'Number of sequences: {len(sequence_len_list)}\n')
        f.write(f'\nTags: \n')

        for tag, count in num_words_per_tag.items():
            f.write(f'{tag}\t{(count / total_words) * 100}%\n')


def main():
    args = arg_parser()
    num_words_per_tag, sequence_len_list = create_tsv_and_fetch_info(input_file=args.input_file, out_dir=args.output_dir)
    write_info_to_file(out_dir=args.output_dir, num_words_per_tag=num_words_per_tag, sequence_len_list=sequence_len_list)

if __name__ == '__main__':
    main()
