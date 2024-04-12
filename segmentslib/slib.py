

def cmp(a, b):
	return (a > b) - (a < b)

def point_orientation(x0, y0, x1, y1, x2, y2):
	return cmp((x2 - x1) * (y1 - y0), (x1 - x0) * (y2 - y1))

def segments_intersect(
	x0, y0, x1, y1,
	x2, y2, x3, y3,
):
	'''
	Given endpoints of two line segments, return whether the segments intersect each other.  Always checks endpoints for intersection.
	'''
	return point_orientation(x0, y0, x1, y1, x2, y2) != point_orientation(x0, y0, x1, y1, x3, y3) and point_orientation(x2, y2, x3, y3, x0, y0) != point_orientation(x2, y2, x3, y3, x1, y1)

def line_intersection_point(
	x0, y0, x1, y1,
	x2, y2, x3, y3,
):
	'''
	Given two pairs of points {(x0, y0), (x1, y1)} and {(x2, y2), (x3, y3)} which define two lines, return the point of intersection between the two lines, or return None if there is no intersection.
	'''
	# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
	D = (x0-x1)*(y2-y3)+(y1-y0)*(x2-x3)
	if D == 0:
		return None
	D = 1./D
	A = (x0*y1-y0*x1)*D
	B = (x2*y3-y2*x3)*D
	return A*(x2-x3)-B*(x0-x1), A*(y2-y3)-B*(y0-y1)


def reflect_vector(vx, vy, rx, ry):
	'''
	Reflect vector (vx, vy) across vector (rx, ry).
	'''
	s = (rx*rx + ry*ry) ** -.5
	nx = ry * s
	ny = -rx * s
	d = 2 * (vx * nx + vy * ny)
	return vx - d * nx, vy - d * ny


def reflect_segment(
	x0, y0, x1, y1,
	x2, y2, x3, y3,
):
	'''
	The segment defined by {(x0, y0), (x1, y1)} is split into two smaller segments at the point of intersection P between {(x0, y0), (x1, y1)} and {(x2, y2), (x3, y3)}.
	Then, reflect the segment {P, (x1, y1)} across {(x2, y2), (x3, y3)} and denote it {P, Q}.
	Return the points P, Q as a flattened tuple (four floats in a single-dimensional tuple).
	'''
	Px, Py = line_intersection_point(x0, y0, x1, y1, x2, y2, x3, y3)
	rx, ry = reflect_vector(x1 - Px, y1 - Py, x3 - x2, y3 - y2)
	return Px, Py, Px + rx, Py + ry


def rectangle_contains_point(rect_x0, rect_y0, rect_x1, rect_y1, point_x, point_y):
	'''
	Return whether point (point_x, point_y) is inside the rectangle that has a minimal corner of (rect_x0, rect_y0) and a maximal corner of (rect_x1, rect_y1).
	'''
	return point_x >= rect_x0 and point_x <= rect_x1 and point_y >= rect_y0 and point_y <= rect_y1

