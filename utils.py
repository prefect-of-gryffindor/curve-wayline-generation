from typing import List
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, polygonize
from scipy import interpolate


def calc_average_distance(x: List[float], y: List[float]):
    """
    calculating d_s
    Returns: average distance of control points in a curve path
    """
    total_length = 0
    for i in range(len(x) - 1):
        total_length += ((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2) ** 0.5
    average_distance = total_length / (len(x) - 1)
    return average_distance


def line_extension(x: List[float], y: List[float], length: float):
    """
    extending both ends of the input line by a certain length
    :param x: x coordinates
    :param y: y coordinates
    :param length: extension length
    :return: x,y coordinates of the extended line
    """
    len_s = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
    len_e = ((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2) ** 0.5
    sx = x[0] + length / len_s * (x[0] - x[1])
    sy = y[0] + length / len_s * (y[0] - y[1])
    ex = x[-1] + length / len_e * (x[-1] - x[-2])
    ey = y[-1] + length / len_e * (y[-1] - y[-2])
    extended_x = [sx] + x + [ex]
    extended_y = [sy] + y + [ey]
    return extended_x, extended_y


def generate_cubic_Bspline_approxi(x: List[float], y: List[float], num: int):
    """
    generating cubic B-spline curve——approximate: the curve does not necessarily pass the points.
    :param x: x coordinates
    :param y: y coordinates
    :param num: the numbers of characteristic points to be output on curve
    :return:x,y coordinates of characteristic points on the generated cubic B-spline curve
    """
    tck, u_origin = interpolate.splprep([x, y], k=3)
    u_num = np.linspace(0, 1, num=num, endpoint=True)
    spline = interpolate.splev(u_num, tck)
    spline_x = spline[0]
    spline_y = spline[1]
    return spline_x, spline_y, u_num, tck


def generate_cubic_Bspline(x: List[float], y: List[float], num: int):
    """
    generating cubic B-spline curve——interpolate: the curve must pass the points
    :param x: x coordinates
    :param y: y coordinates
    :param num: the numbers of characteristic points to be output on curve
    :return:x,y coordinates of characteristic points on the generated cubic B-spline curve
    """
    tck, u_origin = interpolate.splprep([x, y], k=3, s=0)
    u_num = np.linspace(0, 1, num=num, endpoint=True)
    spline = interpolate.splev(u_num, tck)
    spline_x = spline[0]
    spline_y = spline[1]
    return spline_x, spline_y, u_num, tck


def calc_unit_normal_vector(u, tck, **kwargs):
    """
    calculating the unit normal vectors of the waypoints on the wayline (cubic B-spline curve)
    :param u: list, t values corresponding to the characteristic points
    :param tck: tuple, containing the vector of knots, the B-spline coefficients, and the degree of the spline.
    :return: the x,y components of the unit normal vector
    """
    spline = interpolate.splev(u, tck)
    first_derivative = interpolate.splev(u, tck, 1)
    unit_normal_vector_x = []
    unit_normal_vector_y = []
    for i in range(len(first_derivative[0])):
        k_x = first_derivative[0][i] / (first_derivative[0][i] ** 2 + first_derivative[1][i] ** 2) ** 0.5
        k_y = first_derivative[1][i] / (first_derivative[0][i] ** 2 + first_derivative[1][i] ** 2) ** 0.5
        vector_x = -k_y
        vector_y = k_x
        if i != (len(first_derivative[0]) - 1):
            cross_product = (spline[0][i] - spline[0][i + 1]) * vector_y - (spline[1][i] - spline[1][i + 1]) * vector_x
            if cross_product > 0:
                if kwargs['direction'] == 'right':
                    unit_normal_vector_x.append(-vector_x)
                    unit_normal_vector_y.append(-vector_y)
                elif kwargs['direction'] == 'left':
                    unit_normal_vector_x.append(vector_x)
                    unit_normal_vector_y.append(vector_y)
            else:
                if kwargs['direction'] == 'right':
                    unit_normal_vector_x.append(vector_x)
                    unit_normal_vector_y.append(vector_y)
                elif kwargs['direction'] == 'left':
                    unit_normal_vector_x.append(-vector_x)
                    unit_normal_vector_y.append(-vector_y)
        else:
            cross_product = (spline[0][i] - spline[0][i - 1]) * vector_y - (spline[1][i] - spline[1][i - 1]) * vector_x
            if cross_product < 0:
                if kwargs['direction'] == 'right':
                    unit_normal_vector_x.append(-vector_x)
                    unit_normal_vector_y.append(-vector_y)
                elif kwargs['direction'] == 'left':
                    unit_normal_vector_x.append(vector_x)
                    unit_normal_vector_y.append(vector_y)
            else:
                if kwargs['direction'] == 'right':
                    unit_normal_vector_x.append(vector_x)
                    unit_normal_vector_y.append(vector_y)
                elif kwargs['direction'] == 'left':
                    unit_normal_vector_x.append(-vector_x)
                    unit_normal_vector_y.append(-vector_y)
    return unit_normal_vector_x, unit_normal_vector_y


def calc_curvature(u, tck):
    """
    calculating the curvature of the waypoints the wayline (cubic B-spline)
    :param u: list, t values corresponding to the characteristic points
    :param tck: tuple, containing the vector of knots, the B-spline coefficients, and the degree of the spline.
    :return: the curvature of every waypoints on the wayline
    """
    first_derivative = interpolate.splev(u, tck, 1)
    second_derivative = interpolate.splev(u, tck, 2)
    curvature = []
    for i in range(len(u)):
        curvature.append(
            abs(second_derivative[1][i] * first_derivative[0][i] - second_derivative[0][i] * first_derivative[1][i])
            / ((first_derivative[0][i] ** 2 + first_derivative[1][i] ** 2) ** 1.5))
    return curvature


def rm_self_intersection(x: List[float], y: List[float]):
    """
    removing the self-intersecting section on the wayline
    :param x: x coordinates of the wayline
    :param y: y coordinates of the wayline
    :return: a wayline without self-intersection
    """
    line_close = LineString([(x[i], y[i]) for i in range(len(x))] + [(x[0], y[0])])
    multi_line_close = unary_union(line_close)
    multi_poly = polygonize(multi_line_close)
    poly_xy = []
    poly_area = []
    for poly in multi_poly:
        poly_xy.append(list(poly.coords))
        poly_area.append(poly.area)
    target_poly_xy = poly_xy[poly_area.index(max(poly_area))]
    target_poly_xy.pop()
    target_x = [i[0] for i in target_poly_xy]
    target_y = [i[1] for i in target_poly_xy]
    return target_x, target_y


def polygon_clip_line(lx: List[float], ly: List[float], px: List[float], py: List[float]):
    """
    根据多边形边界裁剪条带，返回裁剪后的条带节点集
    clipping the wayline according to the field boundary
    :param lx: x coordinates of the wayline
    :param ly: y coordinates of the wayline
    :param px: x coordinates of the field boundary
    :param py: y coordinates of the field boundary
    :return: x,y coordinates of the clipped wayline
    """
    line = [[lx[i], ly[i]] for i in range(len(lx))]
    polygon = [[px[i], py[i]] for i in range(len(px))]
    origin_line = LineString(line)
    polygon = Polygon(polygon)
    clipped_line = origin_line.intersection(polygon)
    clipped_lx = []
    clipped_ly = []
    if str(type(clipped_line)) != "<class 'shapely.geometry.multilinestring.MultiLineString'>":
        clipped_line = list(clipped_line.coords)
        if len(clipped_line) != 0:
            clipped_lx = [i[0] for i in clipped_line]
            clipped_ly = [i[1] for i in clipped_line]
    else:
        clipped_line_list = list(clipped_line.geoms)
        for line in clipped_line_list:
            clipped_lx.append([list(line.coords)[i][0] for i in range(len(list(line.coords)))])
            clipped_ly.append([list(line.coords)[i][1] for i in range(len(list(line.coords)))])
    return clipped_lx, clipped_ly

