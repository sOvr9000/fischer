

symbols_list_upper = [chr(n) for n in range(913, 930)] + [chr(n) for n in range(931, 938)]
symbols_list_lower = [chr(n) for n in range(945, 970)]
symbols_list = symbols_list_upper + symbols_list_lower

symbols_unreserved_in_math = symbols_list_lower[:]
symbols_unreserved_in_math.remove(chr(959)) # remove omicron from the list to avoid potential confusion with the Latin letter 'o'
symbols_unreserved_in_math.remove(chr(960)) # remove π from the list to avoid potential confusion in numerical contexts

