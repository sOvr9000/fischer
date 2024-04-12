
import cmath as _cmath



__all__ = [
	'primordial_fish', 'owl_donkey', 'galaga_ship', 'rings',
	'shining', 'portrait', 'chevron_balance', 'encased_skull',
	'bulb', 'bowtie', 'crescent', 'crescent_fragments',
	'black_holes', 'stare', 'thorns', 'chaos_transfer',
	'lucky_worlds', 'wings', 'shells', 'gemstones',
	'plating', 'prisms', 'spokes', 'infinity_loops',
	'nautilus', 'pyramid', 'pyramid_sheets', 'shrine',
	'spearhead', 'spearhead_sectioned', 'spearhead_rimmed',
]



def primordial_fish(z):
	return _cmath.log(_cmath.cos(z)) + _cmath.log(z) - z**0.994

def owl_donkey(z):
	return _cmath.log(_cmath.cos(z)) + _cmath.log(z) - z**0.806

def galaga_ship(z):
	return _cmath.log(_cmath.cos(z)) + _cmath.log(z) - 1

def rings(z):
	return _cmath.log(_cmath.cos(z)) + _cmath.log(z) + 0.618**z

def shining(z):
	return _cmath.log(_cmath.cos(z)) + _cmath.log(z) + 1

def portrait(z):
	return _cmath.log(_cmath.cos(z)) + _cmath.log(z) - z**0.17 - z**-0.04 - z**0.4

def chevron_balance(z):
	k = _cmath.log(z) - z**0.994
	return 0.437 * k + 0.307 * (_cmath.log(_cmath.cos(z)) + k)

def encased_skull(z):
	k = _cmath.log(z) - z**0.994
	return 0.49985 * (0.437 * k + 0.307 * (_cmath.log(_cmath.cos(z)) + k)) + 0.315 * galaga_ship(z)

def bulb(z):
	return _cmath.log(_cmath.cos(z)) + _cmath.log(z) + 0.5 * (z**0.5 + 0.86 * (z**0.5 + 0.5 * z**0.2)) + complex(0.5 * (int(z.imag>0)*2-1), 0.4 * (int(z.real>0)*2-1))

def bowtie(z):
	return _cmath.log(_cmath.cos(z)) + _cmath.log(z) + 0.33 * (z**0.35 + 3.5 * (z**-1.2 + 0.34 * z**0.75)) + complex(0.15 * (int(z.imag>0)*2-1), 0.01 * (int(z.real>0)*2-1))

def crescent(z):
	return primordial_fish(z) * 0.025 + chevron_balance(1-shining(1-z)) * 0.975

def crescent_fragments(z):
	return z+0.95194*(crescent(z)-z) # slight continuous-ification, causing mild fragmentation

def black_holes(z):
	return crescent(crescent(0.48161440960640427**z))

def stare(z):
	return crescent(crescent(0.490920613742495**z))

def thorns(z):
	return crescent(crescent(0.5002268178785858**z))

def chaos_transfer(z):
	return crescent(crescent(0.5074316210807205**z))

def lucky_worlds(z):
	return crescent(crescent(0.5116344229486325**z))

def wings(z):
	return crescent(crescent(0.7638025350233489**z))

def shells(z):
	return crescent(crescent(0.7800133422281521**z))

def gemstones(z):
	return crescent(crescent(0.7803135423615744**z))

def plating(z):
	return crescent(crescent(0.8154369579719813**z))

def prisms(z):
	return crescent(crescent(0.8634689793195465**z))

def spokes(z):
	return crescent(crescent(0.8697731821214143**z))

def infinity_loops(z):
	return crescent(crescent(0.8784789859906605**z))

def nautilus(z):
	return crescent(crescent(0.8919879919946632**z))

def pyramid(z):
	return crescent(crescent(0.8967911941294197**z))

def pyramid_sheets(z):
	return crescent(crescent(0.9045963975983990**z))

def shrine(z):
	return shining(shining(z + 0.9767225 ** _cmath.log(z)))

def spearhead(z):
	return nautilus(z + 0.075 ** chaos_transfer(z))

def spearhead_sectioned(z):
	return nautilus(z + 0.13 ** chaos_transfer(z))

def spearhead_rimmed(z):
	return nautilus(z + 0.1 ** chaos_transfer(z))


