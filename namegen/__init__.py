
from random import randint, random, choice


START_CONSONANTS_STANDARD = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'x', 'z']
VOWELS_STANDARD = ['a', 'e', 'i', 'o', 'u']
VOWELS_USE_Y = VOWELS_STANDARD + ['y']
CONSONANTS_STANDARD = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']
CONSONANTS_USE_Y = CONSONANTS_STANDARD + ['y']
DOUBLE_CONSONANTS_STANDARD = [
	'bb', 'cc', 'dd', 'ff', 'gg', 'kk', 'll',
	'mm', 'nn', 'pp', 'rr', 'ss', 'tt', 'zz',

	'bc', 'bd', 'bf', 'bg', 'bh', 'bj', 'bk', 'bl', 'br', 'bs', 'bv', 'bw', 'bz',
	'ch', 'ck', 'cl', 'cr', 'cs', 'ct', 'cz',
	'db', 'df', 'dg', 'dh', 'dk', 'dl', 'dm', 'dn', 'dp', 'dr', 'ds', 'dv', 'dw',
	'fb', 'fd', 'fg', 'fh', 'fk', 'fl', 'fm', 'fn', 'fp', 'fr', 'fs', 'ft', 'fw', 'fz',
	'gb', 'gd', 'gh', 'gl', 'gm', 'gn', 'gr', 'gs',
	'hl', 'hm', 'hn', 'hr', 'hs',
	'kb', 'kd', 'kf', 'kh', 'kl', 'km', 'kn', 'kp', 'kr', 'ks', 'kv', 'kz',
	'lb', 'lc', 'ld', 'lf', 'lg', 'lk', 'lm', 'ln', 'lp', 'lr', 'ls', 'lt', 'lv', 'lx', 'lz',
	'mb', 'md', 'mf', 'mh', 'mn', 'mp', 'mr', 'ms', 'mv', 'mw', 'mz',
	'nc', 'nd', 'nf', 'ng', 'nh', 'nj', 'nk', 'nl', 'nm', 'nr', 'ns', 'nt', 'nv', 'nw', 'nx', 'nz',
	'pf', 'ph', 'pl', 'pn', 'pr', 'ps', 'pt', 'pz',
	'rb', 'rc', 'rd', 'rf', 'rg', 'rh', 'rj', 'rk', 'rl', 'rm', 'rn', 'rp', 'rs', 'rt', 'rv', 'rx', 'rz',
	'sc', 'sh', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sv', 'sw',
	'th', 'tr', 'ts', 'tv', 'tw', 'tz',
	'vd', 'vh', 'vk', 'vl', 'vn', 'vr',
	'wd', 'wf', 'wg', 'wh', 'wk', 'wl', 'wm', 'wn', 'wp', 'wr', 'ws', 'wt', 'wv', 'wz',
	'xb', 'xc', 'xh', 'xl', 'xm', 'xn', 'xr',
	'zb', 'zd', 'zf', 'zh', 'zk', 'zl', 'zm', 'zn', 'zr',
]
DOUBLE_VOWELS_STANDARD = [
	'aa', 'ee', 'oo',

	'ae', 'ai', 'ao', 'au',
	'ea', 'ei', 'eo', 'eu',
	'ia', 'ie', 'io', 'iu',
	'oa', 'oe', 'oi', 'ou',
	'ua', 'ue', 'ui', 'uo',
]
DOUBLE_VOWELS_USE_Y = DOUBLE_VOWELS_STANDARD + [
	'ay', 'ey', 'iy', 'oy', 'uy',
	'ya', 'ye', 'yi', 'yo', 'yu',
]


def generate_name(
	length = None,
	start_with_vowel = None,
	double_vowel_rate = 0.2,
	double_consonant_rate = 0.2,
	vowels = VOWELS_STANDARD,
	consonants = CONSONANTS_STANDARD,
	double_vowels = DOUBLE_VOWELS_STANDARD,
	double_consonants = DOUBLE_CONSONANTS_STANDARD,
	start_consonants = START_CONSONANTS_STANDARD,
	can_start_with_double = False,
	can_end_with_double = False,
):
	if length is None:
		length = randint(4, 8)
	if start_with_vowel is None:
		start_with_vowel = random() < 0.3
	if start_with_vowel:
		name = choice(vowels)
		if can_start_with_double:
			name += choice(vowels)
		v = True
	else:
		name = choice(start_consonants)
		if can_start_with_double:
			name += choice(consonants)
		v = False
	_g = 1 if can_end_with_double else 2
	while len(name) < length:
		if v:
			v = False
			if len(name) + _g < length and random() < double_consonant_rate:
				name += choice(double_consonants)
			else:
				name += choice(consonants)
		else:
			v = True
			if len(name) + _g < length and random() < double_vowel_rate:
				name += choice(double_vowels)
			else:
				name += choice(vowels)
	return name


if __name__ == '__main__':
	print('\n'.join([
		generate_name()
		for _ in range(100)
	]))


# cool names generated:
# ijucurja
# zossakir
# ulandiba


