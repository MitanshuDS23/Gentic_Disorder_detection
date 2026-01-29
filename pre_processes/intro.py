import os
import random
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

INPUT_VCF = "data/raw/mit.vcf"
OUTPUT_VCF = "data/raw/mit_big.vcf"

MULTIPLIER = 110      # how many times to copy per chromosome
POS_OFFSET_STEP = 1000 # shift positions so they don't collide

random.seed(42)

# ============================================================
# LOAD VCF
# ============================================================

def load_vcf(path):

    header = []
    chrom_map = defaultdict(list)

    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                header.append(line)
            else:
                parts = line.rstrip("\n").split("\t")
                chrom_map[parts[0]].append(parts)

    return header, chrom_map


# ============================================================
# EXPAND
# ============================================================

def expand_variants(chrom_map):

    expanded = []

    for chrom, rows in chrom_map.items():

        for i in range(MULTIPLIER):

            offset = i * POS_OFFSET_STEP

            for parts in rows:

                new_parts = parts.copy()

                try:
                    new_parts[1] = str(int(parts[1]) + offset)
                except:
                    pass

                expanded.append(new_parts)

    return expanded


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading VCF...")
    header, chrom_map = load_vcf(INPUT_VCF)

    total_orig = sum(len(v) for v in chrom_map.values())
    print("Original variants:", total_orig)

    print("Expanding...")
    expanded = expand_variants(chrom_map)

    print("Final variants:", len(expanded))

    with open(OUTPUT_VCF, "w") as f:

        for h in header:
            f.write(h)

        for row in expanded:
            f.write("\t".join(row) + "\n")

    print("\n✅ Large VCF generated!")
    print("Output:", OUTPUT_VCF)


if __name__ == "__main__":
    main()
