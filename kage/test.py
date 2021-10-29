# ====== Bad ==========:

# Function that takes a sequence as argument and returns the reverse complement
def run(seq):
    other_seq = ""
    # Iterate every base
    for base in seq:
        # Convert it to its complement
        other_seq += complement(base)

    # return the reverse
    return other_seq[::-1]


# My dna sequence
a = "ACATAGACATTA"
# Running function to take reverse complement
b = run(a)


# ======== Good ==========:
def reverse_complement(sequence):
    reverse_complement_sequence = ""
    for base in sequence:
        reverse_complement_sequence += complement(base)

    return reverse_complement_sequence[::-1]  # This reverses the string


sequence = "AACACATACA"
reverse_complement_of_sequence = reverse_complement(sequence)



