# from scipy.spatial import distance

# a = "1111100001001011010000110111101000100110111110111110110010111100"
# b = "1101111001100101010110100101011000100110111110101011001111111100"

# a_list = a.split()
# b_list = b.split()

# a_list = list(map(int, a))
# b_list = list(map(int, b))

# print(a_list)
# print(b_list)

# print(distance.hamming(a_list, b_list) * len(a))


def popcnt1(n):
    return bin(n).count("1")

a = 16231293029110483773
b = 11868234515962826949
c = 12005043126634442064
d = 2383961605420246012


print(popcnt1(a^c))
print(popcnt1(b^d))

