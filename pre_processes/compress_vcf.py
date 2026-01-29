import os
import argparse
from collections import defaultdict


def compress_vcf(input_vcf, output_vcf, keep_ratio=5):
    """
    Keeps 1 out of `keep_ratio` rows per chromosome.
    """

    chrom_counters = defaultdict(int)

    with open(input_vcf, "r") as fin, open(output_vcf, "w") as fout:

        for line in fin:

            # Keep headers always
            if line.startswith("#"):
                fout.write(line)
                continue

            parts = line.strip().split("\t")
            chrom = parts[0]

            chrom_counters[chrom] += 1

            # Keep 1 of every N rows
            if chrom_counters[chrom] % keep_ratio == 0:
                fout.write(line)


    print("✅ Compression done!")
    print(f"Input : {input_vcf}")
    print(f"Output: {output_vcf}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Downsample VCF per chromosome")

    parser.add_argument(
        "--input",
        required=True,
        help="Input VCF path"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output VCF path"
    )

    parser.add_argument(
        "--ratio",
        type=int,
        default=5,
        help="Keep 1 of every N rows per chromosome (default=5)"
    )

    args = parser.parse_args()

    compress_vcf(args.input, args.output, args.ratio)
